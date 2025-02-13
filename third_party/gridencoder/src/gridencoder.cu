#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF... program will never reach here!
 __device__ inline at::Half atomicAdd(at::Half *address, at::Half val) {
  // requires CUDA >= 10 and ARCH >= 70
  // this is very slow compared to float or __half2, never use it.
  //return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename T>
__device__ inline T smoothstep(T val) {
	return val*val*(3.0f - 2.0f * val);
}

template <typename T>
__device__ inline T smoothstep_derivative(T val) {
	return 6*val*(1.0f - val);
}


template <uint32_t num_dim>
__device__ uint32_t fast_hash(const uint32_t pos_grid[num_dim]) {

    // coherent type of hashing
    constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };

    uint32_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < num_dim; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}


template <uint32_t num_dim, uint32_t n_fearures>
// gridtype, 0, hashmap_size, resolution, pos_grid_local
__device__ uint32_t get_grid_index(const uint32_t gridtype, const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[num_dim]) {
    uint32_t stride = 1;
    uint32_t index = 0;

    #pragma unroll

    // if level is small, hashtable is long enough, then no hash trick is needed
    // final index = pos_grid[0] + pos_grid[1] * resolution + pos_grid[2] * resolution ^ 2
    // This is to get the ordered index (eg: idx = W*H+w for 2D case)
    for (uint32_t d = 0; d < num_dim && stride <= hashmap_size; d++) {
        // pos_grid[0] -> pos_grid[0] + pos_grid[1] * resolution -> pos_grid[0] + pos_grid[1] * resolution + pos_grid[2] * resolution ^ 2
        index += pos_grid[d] * stride;
        // resolution -> resolution^2 -> resolution^3
        stride *= resolution;
    }

    // NOTE: for NeRF, the hash is in fact not necessary. Check https://github.com/NVlabs/instant-ngp/issues/97.
    // gridtype: 0 == hash, 1 == tiled
    if (gridtype == 0 && stride > hashmap_size) {
        index = fast_hash<num_dim>(pos_grid);
    }

    // (index % hashmap_size) to avoid overflow
    // notice: here n_fearures is multipled
    return (index % hashmap_size) * n_fearures + ch;
}

// kernel_grid<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>
// (inputs, embeddings, offsets_list, outputs, N, n_levels, S, H, dy_dx, gridtype, align_corners, interp);
// N: N_rays
// n_levels: level
// S: log2(per_level_scale)
// H: base_resolution
////// One CPU kernel calls one GPU grid, one GPU grid contains several blocks, one block contains several threads
template <typename scalar_t, uint32_t num_dim, uint32_t n_fearures>  // <scalar_t, num_dim, 2 // num_dim: coords input_dim = 3
// __global__ means called by CPU and conducted by GPU
// always no return, so always void
__global__ void kernel_grid(
    const float * __restrict__ inputs,  //  has been mapped to [0, 1]  [N, num_dim]
    const scalar_t * __restrict__ grid,  // here grid means hashgrid not processing grid in GPU shape:[offset*n_fearures]?
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ outputs,   // non-constant
    const uint32_t N, const uint32_t n_levels, const uint32_t Rb, const float PV,
    scalar_t * __restrict__ dy_dx,   // non-constant
    const bool * __restrict__ binary_vxl,
    const int * __restrict__ min_level_id  // [N]
    ) {
    // grid > block > thread
    // blockIdx.x is idx of the block along x axis, blockDim.x is th block width, threadIdx.x is idx along the width of thread
    // get the place of corresponding parallel computing point
    // get b: the index of [0, N_rays)
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N_rays axis

    if (b >= N) return;  // deprecate not needed threads

    uint32_t level = 0;

    if (min_level_id) {
        level = min_level_id[b] + blockIdx.y;  // parallel along level axis, current level
        // printf(" A%dB%d", min_level_id[b], level);
    }
    else {
        level = blockIdx.y;
    }

    // locate  why these variables are changed? because they are pointers when defined? --> get the locate of the data to be processed
    grid += (uint32_t)offsets_list[level] * n_fearures;
    inputs += b * num_dim;
    outputs += blockIdx.y * N * n_fearures + b * n_fearures;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < n_fearures; ch++) {  // traverse each feature_dim
            outputs[ch] = 0;
        }
        if (dy_dx) {
            dy_dx += b * num_dim * n_levels * n_fearures + blockIdx.y * num_dim * n_fearures; // N n_levels num_dim n_fearures
            #pragma unroll
            for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
                #pragma unroll
                for (uint32_t ch = 0; ch < n_fearures; ch++) {  // traverse each feature_dim
                    dy_dx[d * n_fearures + ch] = 0;
                }
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    // exp2f(level * S) = 2 ^ (level*S) = 2 ^ (level*log2(per_level_scale)) = 2 ^ (log2(per_level_scale)*level) = per_level_scale ^ level
    // const uint32_t resolution = (uint32_t)ceil(exp2f(level * S) * H);
    const uint32_t resolution = (uint32_t)resolutions_list[level];
    /*  resolution = 6:
          (n,0)->|       |<-(n,1)
                0 0 0 0 0 0
                0 1 1 1 1 0
                0 1 1 1 1 0
                0 1 1 1 1 0
                0 1 1 1 1 0
                0 0 0 0 0 0
    */

    // calculate coordinate (always use float for precision!)
    float pos[num_dim];
    float pos_deriv[num_dim];
    uint32_t pos_grid[num_dim];

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim

        pos[d] = inputs[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        pos_deriv[d] = 1.0f;
    }

    // verification of alignment
    // if (level == n_levels - 1 && b < 4) {
    //     printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);
    // }

    // interpolate
    scalar_t results[n_fearures] = {0}; // temp results in register
    float w_list[1 << num_dim] = {0};
    bool m_list[1 << num_dim] = {true};
    uint32_t zero_flag_list[1 << num_dim] = {0};
    uint32_t index_list[1 << num_dim] = {0};
    float wn = 0;

    /*
    loop 0:
    idx = 0 = 000, w = 1,
    d = 0, 1<<d = 001, get into if   -> w=(1-pos[0]),                     pos_grid_local[0] = pos_grid[0]
    d = 1, 1<<d = 010, get into if   -> w=(1-pos[0])(1-pos[1]),           pos_grid_local[1] = pos_grid[1]
    d = 2, 1<<d = 100, get into if   -> w=(1-pos[0])(1-pos[1])(1-pos[2]), pos_grid_local[2] = pos_grid[2]
    loop 1:
    idx = 1 = 001, w = 1,
    d = 0, 1<<d = 001, get into else -> w=(pos[0]),                       pos_grid_local[0] = pos_grid[0]+1
    d = 1, 1<<d = 010, get into if   -> w=(pos[0])(1-pos[1]),             pos_grid_local[1] = pos_grid[1]
    d = 2, 1<<d = 100, get into if   -> w=(pos[0])(1-pos[1])(1-pos[2]),   pos_grid_local[2] = pos_grid[2]
    loop 2:
    idx = 2 = 010, w = 1,
    d = 0, 1<<d = 001, get into if   -> w=(1-pos[0]),                     pos_grid_local[0] = pos_grid[0]
    d = 1, 1<<d = 010, get into else -> w=(1-pos[0])(pos[1]),             pos_grid_local[1] = pos_grid[1]+1
    d = 2, 1<<d = 100, get into if   -> w=(1-pos[0])(pos[1])(1-pos[2]),   pos_grid_local[2] = pos_grid[2]
    loop 3:
    idx = 3 = 011, w = 1,
    d = 0, 1<<d = 001, get into else -> w=(pos[0]),                       pos_grid_local[0] = pos_grid[0]+1
    d = 1, 1<<d = 010, get into else -> w=(pos[0])(pos[1]),               pos_grid_local[1] = pos_grid[1]+1
    d = 2, 1<<d = 100, get into if   -> w=(pos[0])(pos[1])(1-pos[2]),     pos_grid_local[2] = pos_grid[2]
    ...
    rule: idx=100 from back to front:001 -> 0:w*=(1-pos), 1:w*=pos; 0:pos_grid_local=pos_grid, 1:pos_grid_local=pos_grid+1
    */

    /* a toy 2D example for w important
    f1          f2
        (v)
    (u)  f  (1-u)
       (1-v)

    f3          f4

    get: f = uvf4 + (1-u)vf3 + u(1-v)f2 + (1-u)(1-v)f1
    */
    #pragma unroll
    // idx = {0, 1, 2, 3, 4, 5, 6, 7}
    // here loop_num==8 is because there are 8 vertextes for interp 8=2**num_dim
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) { // why parallel is not applied for this loop?
        float w = 1;  // define weight for triblinear interp for different vertexes
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {  // (1 << d) = {1, 2, 4}
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }

        // gridtype: "hash" hashmap_size: hash map size of current level

        uint32_t zero_flag = 0;
        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                zero_flag = 1;
                break;
            }
        }

        bool m = true;
        if (binary_vxl){
            m = false;
            float scale_re = 1.0 / (float(resolution) - 2.0);
            uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
            #pragma unroll
            for (uint32_t d = 0; d < num_dim; d++) {
                float points_n = (float(pos_grid_local[d]) - 0.5) * scale_re;

                float pos_g1 = points_n - scale_re;
                pos_g1 = pos_g1 * Rb;
                pos_g1 = pos_g1 < 0? 0:pos_g1;
                pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
                pos_g[d] = int(pos_g1);

                float pos_g2 = points_n + scale_re;
                pos_g2 = pos_g2 * Rb;
                pos_g2 = pos_g2 < 0? 0:pos_g2;
                pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
                pos_g[num_dim + d] = int(pos_g2);
            }

            if (num_dim == 1) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[1]; idx_a++){
                    m = m | (binary_vxl[idx_a]);
                    if (m == true) break;
                }
            }
            if (num_dim == 2) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
                        m = m | (binary_vxl[idx_a*Rb+idx_b]);
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
            if (num_dim == 3) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[3]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[4]; idx_b++){
                        #pragma unroll
                        for (int idx_c=pos_g[2]; idx_c<=pos_g[5]; idx_c++){
                            m = m | (binary_vxl[idx_a*Rb*Rb+idx_b*Rb+idx_c]);
                            if (m == true) break;
                        }
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
        }

        w_list[idx] = w;
        m_list[idx] = m;
        zero_flag_list[idx] = zero_flag;
        if (zero_flag == 0 && m == true){
            uint32_t index = get_grid_index<num_dim, n_fearures>(0, 0, hashmap_size, resolution, pos_grid_local);
            index_list[idx] = index;
            wn += w;
        }
    }

    if (wn == 0) {
        wn += 1e-9;
    }
    float wn_re = 1.0 / wn;

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {
        // note: m_list is designed for 3D grid context interpolation.
        // for normal point interpolation, m_list should always be all true.
        if (zero_flag_list[idx] == 0 && m_list[idx] == true) {  // valid point
            // writing to register (fast)
            #pragma unroll
            for (uint32_t ch = 0; ch < n_fearures; ch++) {
                results[ch] += w_list[idx] * wn_re * grid[index_list[idx] + ch];  // index is already multipled by n_fearures.
            }
        }
        /* else {
            for (uint32_t ch = 0; ch < n_fearures; ch++) {
                results[ch] += PV;  // index is already multipled by n_fearures.
            }
        } */
        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < n_fearures; ch++) {
        outputs[ch] = results[ch];
    }

    // prepare dy_dx
    // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
    if (dy_dx) {

        // level: current level
        // n_levels: max level
        dy_dx += b * num_dim * n_levels * n_fearures + blockIdx.y * num_dim * n_fearures; // N n_levels num_dim n_fearures

        /*  12 edges

        big loop 001: gd=0  // 0,1,2
            idx=0=000  // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd>=gd -> d=nd+1=1
                    1<<nd=000. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[1])
                        pos_grid_local[1] = pos_grid[1]
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[1])*(1 - pos[2])
                        pos_grid_local[2] = pos_grid[2]
                pos_grid_local[0] = pos_grid[0]

                pos_grid_local = [pg[0], pg[1], pg[2]]
                pos_grid_local = [pg[0]+1, pg[1], pg[2]]

            idx=1=001 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd>=gd -> d=nd+1=1
                    1<<nd=000. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[1]
                        pos_grid_local[1] = pos_grid[1]+1
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 True
                        w = resolution*pos[1]*(1 - pos[2])
                        pos_grid_local[2] = pos_grid[2]
                pos_grid_local[0] = pos_grid[0]

                pos_grid_local = [pg[0], pg[1]+1, pg[2]]
                pos_grid_local = [pg[0]+1, pg[1]+1, pg[2]]

            idx=2=010 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd>=gd -> d=nd+1=1
                    1<<nd=000. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[1])
                        pos_grid_local[1] = pos_grid[1]
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 False
                        w = resolution*(1 - pos[1])*pos[2]
                        pos_grid_local[2] = pos_grid[2]+1
                pos_grid_local[0] = pos_grid[0]

                pos_grid_local = [pg[0], pg[1], pg[2]+1]
                pos_grid_local = [pg[0]+1, pg[1], pg[2]+1]

            idx=3=011 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd>=gd -> d=nd+1=1
                    1<<nd=000. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[1]
                        pos_grid_local[1] = pos_grid[1]+1
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[1]*pos[2]
                        pos_grid_local[2] = pos_grid[2]+1
                pos_grid_local[0] = pos_grid[0]

                pos_grid_local = [pg[0], pg[1]+1, pg[2]+1]
                pos_grid_local = [pg[0]+1, pg[1]+1, pg[2]+1]

            results_grad = {resolution*(1 - pos[1])*(1 - pos[2])} * {grid([pg[0]+1, pg[1], pg[2]])-grid([pg[0], pg[1], pg[2]])} + \
                           {resolution*pos[1]*(1 - pos[2])} * {grid([pg[0]+1, pg[1]+1, pg[2]])-grid([pg[0], pg[1]+1, pg[2]])} + \
                           {resolution*(1 - pos[1])*pos[2]} * {grid([pg[0]+1, pg[1], pg[2]+1])-grid([pg[0], pg[1], pg[2]+1])} + \
                           {resolution*pos[1]*pos[2]} * {grid([pg[0]+1, pg[1]+1, pg[2]+1])-grid([pg[0], pg[1]+1, pg[2]+1])} + \

        big loop 001: gd=1  // 0,1,2
            idx=0=000  // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd<gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[0])
                        pos_grid_local[0] = pos_grid[0]
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[0])*(1 - pos[2])
                        pos_grid_local[2] = pos_grid[2]
                pos_grid_local[1] = pos_grid[1]

                pos_grid_local = [pg[0], pg[1], pg[2]]
                pos_grid_local = [pg[0], pg[1]+1, pg[2]]

            idx=1=001 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd<gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[0]
                        pos_grid_local[0] = pos_grid[0]+1
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 True
                        w = resolution*pos[1]*(1 - pos[2])
                        pos_grid_local[2] = pos_grid[2]
                pos_grid_local[1] = pos_grid[1]

                pos_grid_local = [pg[0]+1, pg[1], pg[2]]
                pos_grid_local = [pg[0]+1, pg[1]+1, pg[2]]

            idx=2=010 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd<gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[0])
                        pos_grid_local[0] = pos_grid[0]
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 False
                        w = resolution*(1 - pos[1])*pos[2]
                        pos_grid_local[2] = pos_grid[2]+1
                pos_grid_local[1] = pos_grid[1]

                pos_grid_local = [pg[0], pg[1], pg[2]+1]
                pos_grid_local = [pg[0], pg[1]+1, pg[2]+1]

            idx=3=011 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd>=gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[0]
                        pos_grid_local[0] = pos_grid[0]+1
                nd=1 // 0,1
                    nd>=gd -> d=nd+1=2
                    1<<nd=001. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[1]*pos[2]
                        pos_grid_local[2] = pos_grid[2]+1
                pos_grid_local[1] = pos_grid[1]

                pos_grid_local = [pg[0]+1, pg[1], pg[2]+1]
                pos_grid_local = [pg[0]+1, pg[1]+1, pg[2]+1]


        big loop 010: gd=2  // 0,1,2
            idx=0=000  // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd<gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[0])
                        pos_grid_local[0] = pos_grid[0]
                nd=1 // 0,1
                    nd<gd -> d=nd=1
                    1<<nd=001. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[0])*(1 - pos[1])
                        pos_grid_local[1] = pos_grid[1]
                pos_grid_local[2] = pos_grid[2]

                pos_grid_local = [pg[0], pg[1], pg[2]]
                pos_grid_local = [pg[0], pg[1], pg[2]+1]

            idx=1=001 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd<gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[0]
                        pos_grid_local[0] = pos_grid[0]+1
                nd=1 // 0,1
                    nd<gd -> d=nd=1
                    1<<nd=001. (idx & (1 << nd)) == 0 True
                        w = resolution*pos[1]*(1 - pos[1])
                        pos_grid_local[1] = pos_grid[1]
                pos_grid_local[2] = pos_grid[2]

                pos_grid_local = [pg[0]+1, pg[1], pg[2]]
                pos_grid_local = [pg[0]+1, pg[1], pg[2]+1]

            idx=2=010 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd<gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 True
                        w = resolution*(1 - pos[0])
                        pos_grid_local[0] = pos_grid[0]
                nd=1 // 0,1
                    nd>=gd -> d=nd=1
                    1<<nd=001. (idx & (1 << nd)) == 0 False
                        w = resolution*(1 - pos[1])*pos[1]
                        pos_grid_local[1] = pos_grid[1]+1
                pos_grid_local[2] = pos_grid[2]

                pos_grid_local = [pg[0], pg[1]+1, pg[2]]
                pos_grid_local = [pg[0], pg[1]+1, pg[2]+1]

            idx=3=011 // 0,1,2,3
                w=resolution
                nd=0 // 0,1
                    nd>=gd -> d=nd=0
                    1<<nd=000. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[0]
                        pos_grid_local[0] = pos_grid[0]+1
                nd=1 // 0,1
                    nd>=gd -> d=nd=1
                    1<<nd=001. (idx & (1 << nd)) == 0 False
                        w = resolution*pos[1]*pos[1]
                        pos_grid_local[1] = pos_grid[1]+1
                pos_grid_local[2] = pos_grid[2]

                pos_grid_local = [pg[0]+1, pg[1]+1, pg[2]]
                pos_grid_local = [pg[0]+1, pg[1]+1, pg[2]+1]


        */

        #pragma unroll
        for (uint32_t gd = 0; gd < num_dim; gd++) {  // D是num_dim=3

            scalar_t results_grad[n_fearures] = {0};

            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (num_dim - 1)); idx++) { // 0, 1, 2, 3
                // float w = (float)(align_corners ? resolution - 1 : resolution);
                float w = (float)(resolution - 2);
                uint32_t pos_grid_local[num_dim];

                #pragma unroll
                for (uint32_t nd = 0; nd < num_dim - 1; nd++) {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
                    }
                }

                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left = 0;
                uint32_t zero_flag_left = 0;
                for (uint32_t d = 0; d < num_dim; d++) {
                    if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                        zero_flag_left = 1;
                        break;
                    }
                }
                if (zero_flag_left==0) {
                    index_left = get_grid_index<num_dim, n_fearures>(0, 0, hashmap_size, resolution, pos_grid_local);
                }

                pos_grid_local[gd] = min(pos_grid[gd] + 1, resolution - 1);
                uint32_t index_right = 0;
                uint32_t zero_flag_right = 0;
                for (uint32_t d = 0; d < num_dim; d++) {
                    if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                        zero_flag_right = 1;
                        break;
                    }
                }
                if (zero_flag_right==0) {
                    index_right = get_grid_index<num_dim, n_fearures>(0, 0, hashmap_size, resolution, pos_grid_local);
                }


                #pragma unroll
                for (uint32_t ch = 0; ch < n_fearures; ch++) {
                    // calculate gradient by w
                    float grid_left = 0;
                    float grid_right = 0;
                    if (zero_flag_left==0)
                        grid_left = grid[index_left + ch];
                    if (zero_flag_right==0)
                        grid_right = grid[index_right + ch];

                    results_grad[ch] += w * (grid_right - grid_left) * pos_deriv[gd];
                }
            }

            #pragma unroll
            for (uint32_t ch = 0; ch < n_fearures; ch++) {
                dy_dx[gd * n_fearures + ch] = results_grad[ch];
            }
        }
    }
}


// grad, inputs, embeddings, offsets_list, grad_embeddings, N, num_dim, n_fearures, n_levels, max_level, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation
template <typename scalar_t, uint32_t num_dim, uint32_t n_fearures, uint32_t n_features_per_thread>  // n_features_per_thread is n_features_per_thread
__global__ void kernel_grid_backward(
    const scalar_t * __restrict__ grad,  // grad is the gradient from loss
    const float * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ grad_grid,   // same type as grad
    const uint32_t N, const uint32_t n_levels, const uint32_t Rb,
    const bool * __restrict__ binary_vxl,
    const int * __restrict__ min_level_id
    ) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread / n_fearures;
    if (b >= N) return;

    uint32_t level = 0;

    if (min_level_id) {
        level = min_level_id[b] + blockIdx.y;  // parallel along level axis, current level
    }
    else {
        level = blockIdx.y;
    }

    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread - b * n_fearures;

    // locate
    grad_grid += offsets_list[level] * n_fearures;
    inputs += b * num_dim;
    grad += blockIdx.y * N * n_fearures + b * n_fearures + ch; // n_levels, N, n_fearures

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    // const uint32_t resolution = (uint32_t)ceil(exp2f(level * S) * H);
    const uint32_t resolution = (uint32_t)resolutions_list[level];

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[num_dim];
    uint32_t pos_grid[num_dim];

    // same as forward process
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        pos[d] = inputs[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    scalar_t grad_cur[n_features_per_thread] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < n_features_per_thread; c++) {
        grad_cur[c] = grad[c];
    }

    float w_list[1 << num_dim] = {0};
    bool m_list[1 << num_dim] = {true};
    uint32_t zero_flag_list[1 << num_dim] = {0};
    uint32_t index_list[1 << num_dim] = {0};
    float wn = 0;

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {
        float w = 1;
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }

        uint32_t zero_flag = 0;
        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                zero_flag = 1;
                break;
            }
        }

        bool m = true;
        if (binary_vxl){
            m = false;
            float scale_re = 1.0 / (float(resolution) - 2.0);
            uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
            #pragma unroll
            for (uint32_t d = 0; d < num_dim; d++) {
                float points_n = (float(pos_grid_local[d]) - 0.5) * scale_re;

                float pos_g1 = points_n - scale_re;
                pos_g1 = pos_g1 * Rb;
                pos_g1 = pos_g1 < 0? 0:pos_g1;
                pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
                pos_g[d] = int(pos_g1);

                float pos_g2 = points_n + scale_re;
                pos_g2 = pos_g2 * Rb;
                pos_g2 = pos_g2 < 0? 0:pos_g2;
                pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
                pos_g[num_dim + d] = int(pos_g2);
            }

            if (num_dim == 1) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[1]; idx_a++){
                    m = m | (binary_vxl[idx_a]);
                    if (m == true) break;
                }
            }
            if (num_dim == 2) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
                        m = m | (binary_vxl[idx_a*Rb+idx_b]);
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
            if (num_dim == 3) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[3]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[4]; idx_b++){
                        #pragma unroll
                        for (int idx_c=pos_g[2]; idx_c<=pos_g[5]; idx_c++){
                            m = m | (binary_vxl[idx_a*Rb*Rb+idx_b*Rb+idx_c]);
                            if (m == true) break;
                        }
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
        }

        w_list[idx] = w;
        m_list[idx] = m;
        zero_flag_list[idx] = zero_flag;
        if (zero_flag == 0 && m == true){
            uint32_t index = get_grid_index<num_dim, n_fearures>(0, ch, hashmap_size, resolution, pos_grid_local);
            index_list[idx] = index;
            wn += w;
        }
    }

    if (wn == 0) {
        wn += 1e-9;
    }
    float wn_re = 1.0 / wn;

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {
        if (zero_flag_list[idx] == 0 && m_list[idx] == true) {

            // for atomicAdd, see https://www.cnblogs.com/biglucky/p/4283476.html. It is plus operation

            // atomicAdd for __half is slow (especially for large values), so we use __half2 if n_features_per_thread % 2 == 0
            // TODO: use float which is better than __half, if n_features_per_thread % 2 != 0
            if (std::is_same<scalar_t, at::Half>::value && n_features_per_thread % 2 == 0) {  // in this code it should be in this line
                #pragma unroll
                for (uint32_t c = 0; c < n_features_per_thread; c += 2) {
                    // process two __half at once (by interpreting as a __half2)
                    __half2 v = {(__half)(w_list[idx] * wn_re * grad_cur[c]), (__half)(w_list[idx] * wn_re * grad_cur[c + 1])};
                    atomicAdd((__half2*)&grad_grid[index_list[idx] + c], v);
                }
            // float, or __half when n_features_per_thread % 2 != 0 (which means n_fearures == 1)
            } else {
                #pragma unroll
                for (uint32_t c = 0; c < n_features_per_thread; c++) {
                    atomicAdd(&grad_grid[index_list[idx] + c], w_list[idx] * wn_re * grad_cur[c]);
                }
            }
        }
    }
}


template <typename scalar_t, uint32_t num_dim, uint32_t n_fearures>
__global__ void kernel_input_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,
    scalar_t * __restrict__ grad_inputs,
    uint32_t N, uint32_t n_levels
    ) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= N * num_dim) return;

    const uint32_t b = t / num_dim;
    const uint32_t d = t - b * num_dim;

    dy_dx += b * n_levels * num_dim * n_fearures;

    scalar_t result = 0;

    # pragma unroll
    for (int l = 0; l < n_levels; l++) {
        # pragma unroll
        for (int ch = 0; ch < n_fearures; ch++) {
            result += grad[l * N * n_fearures + b * n_fearures + ch] * dy_dx[l * num_dim * n_fearures + d * n_fearures + ch];
        }
    }

    grad_inputs[t] = result;
}


template <typename scalar_t, uint32_t num_dim>
void kernel_grid_wrapper(
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *outputs,
    const uint32_t N, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb, const float PV,
    scalar_t *dy_dx,
    const bool *binary_vxl,
    const int *min_level_id
    ) {
    // blocks and threads are defined here
    static constexpr uint32_t N_THREAD = 512;
    // div_round_up is (N + N_THREAD - 1) / N_THREAD
    const dim3 blocks_hashgrid = { div_round_up(N, N_THREAD), n_levels, 1 };
    switch (n_fearures) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: kernel_grid<scalar_t, num_dim, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        case 2: kernel_grid<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        case 4: kernel_grid<scalar_t, num_dim, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        case 8: kernel_grid<scalar_t, num_dim, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        case 16: kernel_grid<scalar_t, num_dim, 16><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        case 32: kernel_grid<scalar_t, num_dim, 32><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        default: throw std::runtime_error{"GridEncoding: n_fearures must be 1, 2, 4, 8, 16 or 32."};
    }
}

// inputs: [N, num_dim], float, in [0, 1]
// embeddings: [sO, n_fearures], float
// offsets_list: [n_levels + 1], uint32_t
// outputs: [n_levels, N, n_fearures], float (n_levels first, so only one level of hashmap needs to fit into cache at a time.)
// H: base resolution
// dy_dx: [N, n_levels * num_dim * n_fearures]
template <typename scalar_t>
void grid_encode_forward_cuda(
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *outputs,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb, const float PV,
    scalar_t *dy_dx,
    const bool *binary_vxl,
    const int *min_level_id
    ) {
    switch (num_dim) {
        case 1: kernel_grid_wrapper<scalar_t, 1>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_fearures, n_levels, max_level, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        case 2: kernel_grid_wrapper<scalar_t, 2>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_fearures, n_levels, max_level, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_fearures, n_levels, max_level, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        // case 4: kernel_grid_wrapper<scalar_t, 4>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_fearures, n_levels, max_level, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        // case 5: kernel_grid_wrapper<scalar_t, 5>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_fearures, n_levels, max_level, Rb, PV, dy_dx, binary_vxl, min_level_id); break;
        default: throw std::runtime_error{"GridEncoding: num_dim must be 1, 2, 3."};
    }
}

template <typename scalar_t, uint32_t num_dim>
void kernel_grid_backward_wrapper(
    const scalar_t *grad,
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *grad_embeddings,
    const uint32_t N, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb,
    scalar_t *dy_dx,
    scalar_t *grad_inputs,
    const bool *binary_vxl,
    const int *min_level_id
    ) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t n_features_per_thread = std::min(2u, n_fearures); // n_features_per_thread
    const dim3 blocks_hashgrid = { div_round_up(N * n_fearures / n_features_per_thread, N_THREAD), n_levels, 1 };
    switch (n_fearures) {
        case 1:
            kernel_grid_backward<scalar_t, num_dim, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl, min_level_id);
            if (dy_dx) kernel_input_backward<scalar_t, num_dim, 1><<<div_round_up(N * num_dim, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, N, n_levels);
            break;
        case 2:
            kernel_grid_backward<scalar_t, num_dim, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl, min_level_id);
            if (dy_dx) kernel_input_backward<scalar_t, num_dim, 2><<<div_round_up(N * num_dim, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, N, n_levels);
            break;
        case 4:
            kernel_grid_backward<scalar_t, num_dim, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl, min_level_id);
            if (dy_dx) kernel_input_backward<scalar_t, num_dim, 4><<<div_round_up(N * num_dim, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, N, n_levels);
            break;
        case 8:
            kernel_grid_backward<scalar_t, num_dim, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl, min_level_id);
            if (dy_dx) kernel_input_backward<scalar_t, num_dim, 8><<<div_round_up(N * num_dim, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, N, n_levels);
            break;
        case 16:
            kernel_grid_backward<scalar_t, num_dim, 16, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl, min_level_id);
            if (dy_dx) kernel_input_backward<scalar_t, num_dim, 16><<<div_round_up(N * num_dim, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, N, n_levels);
            break;
        case 32:
            kernel_grid_backward<scalar_t, num_dim, 32, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl, min_level_id);
            if (dy_dx) kernel_input_backward<scalar_t, num_dim, 32><<<div_round_up(N * num_dim, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, N, n_levels);
            break;
        default: throw std::runtime_error{"GridEncoding: n_fearures must be 1, 2, 4, 8, 16 or 32."};
    }
}


// grad: [n_levels, N, n_fearures], float
// inputs: [N, num_dim], float, in [0, 1]
// embeddings: [sO, n_fearures], float
// offsets_list: [n_levels + 1], uint32_t
// grad_embeddings: [sO, n_fearures]
// H: base resolution
template <typename scalar_t>
void grid_encode_backward_cuda(
    const scalar_t *grad,
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *grad_embeddings,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb,
    scalar_t *dy_dx,
    scalar_t *grad_inputs,
    const bool *binary_vxl,
    const int *min_level_id
    ) {
    switch (num_dim) {
        case 1: kernel_grid_backward_wrapper<scalar_t, 1>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_fearures, n_levels, max_level, Rb, dy_dx, grad_inputs, binary_vxl, min_level_id); break;
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_fearures, n_levels, max_level, Rb, dy_dx, grad_inputs, binary_vxl, min_level_id); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_fearures, n_levels, max_level, Rb, dy_dx, grad_inputs, binary_vxl, min_level_id); break;
        // case 4: kernel_grid_backward_wrapper<scalar_t, 4>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_fearures, n_levels, max_level, Rb, dy_dx, grad_inputs, binary_vxl, min_level_id); break;
        // case 5: kernel_grid_backward_wrapper<scalar_t, 5>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_fearures, n_levels, max_level, Rb, dy_dx, grad_inputs, binary_vxl, min_level_id); break;
        default: throw std::runtime_error{"GridEncoding: num_dim must be 1, 2, 3."};
    }
}



void grid_encode_forward(
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor outputs,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb, const float PV,
    at::optional<at::Tensor> dy_dx,
    const at::optional<at::Tensor> binary_vxl,
    const at::optional<at::Tensor> min_level_id
    ) {

    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(outputs);
    // CHECK_CUDA(dy_dx);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(outputs);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(outputs);
    // CHECK_IS_FLOATING(dy_dx);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF is the action to lunch a kernal
    // embeddings.scalar_type() indicates the type of data, to decide the type of lunched kernal
    // "grid_encode_forward" indicates the name in traceback, when reporting error...
    // grid_encode_forward_cuda name of function
    // process: first generate an output (already done), and fill data into it.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "grid_encode_forward",
    ([&] {
        grid_encode_forward_cuda<scalar_t>(
            inputs.data_ptr<float>(),
            embeddings.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            outputs.data_ptr<scalar_t>(),
            N, num_dim, n_fearures, n_levels, max_level, Rb, PV,
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr,
            binary_vxl.has_value() ? binary_vxl.value().data_ptr<bool>() : nullptr,
            min_level_id.has_value() ? min_level_id.value().data_ptr<int>() : nullptr
            );
    })
    );
}

void grid_encode_backward(
    const at::Tensor grad,
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb,
    const at::optional<at::Tensor> dy_dx,
    at::optional<at::Tensor> grad_inputs,
    const at::optional<at::Tensor> binary_vxl,
    const at::optional<at::Tensor> min_level_id
    ) {

    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(grad_embeddings);
    // CHECK_CUDA(dy_dx);
    // CHECK_CUDA(grad_inputs);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(grad_embeddings);
    // CHECK_CONTIGUOUS(dy_dx);
    // CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(grad_embeddings);
    // CHECK_IS_FLOATING(dy_dx);
    // CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "grid_encode_backward", ([&] {
        grid_encode_backward_cuda<scalar_t>(
            grad.data_ptr<scalar_t>(),
            inputs.data_ptr<float>(),
            embeddings.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            grad_embeddings.data_ptr<scalar_t>(),
            N, num_dim, n_fearures, n_levels, max_level, Rb,
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr,
            grad_inputs.has_value() ? grad_inputs.value().data_ptr<scalar_t>() : nullptr,
            binary_vxl.has_value() ? binary_vxl.value().data_ptr<bool>() : nullptr,
            min_level_id.has_value() ? min_level_id.value().data_ptr<int>() : nullptr
            );
    }));

}


///////////////////////////////////////////////////////////////


template <typename scalar_t, uint32_t num_dim, uint32_t n_fearures>  // <scalar_t, num_dim, 2 // num_dim: coords input_dim = 3
// __global__ means called by CPU and conducted by GPU
// always no return, so always void
__global__ void avg_2D_forward_kernel(
    const float * __restrict__ inputs,  //  has been mapped to [0, 1]
    const scalar_t * __restrict__ grid,  // here grid means hashgrid not processing grid in GPU shape:[offset*n_fearures]?
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ outputs,   // non-constant
    const uint32_t N, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const bool * __restrict__ binary_vxl
    ) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N_rays axis

    if (b >= N) return;  // deprecate not needed threads

    const uint32_t level = blockIdx.y;  // parallel along level axis, current level

    // locate  why these variables are changed? because they are pointers when defined? --> get the locate of the data to be processed
    grid += (uint32_t)offsets_list[level] * n_fearures;
    inputs += b * num_dim;
    outputs += level * N * n_fearures + b * n_fearures;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < n_fearures; ch++) {  // traverse each feature_dim
            outputs[ch] = 0;
        }
        return;
    }

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    // exp2f(level * S) = 2 ^ (level*S) = 2 ^ (level*log2(per_level_scale)) = 2 ^ (log2(per_level_scale)*level) = per_level_scale ^ level
    // const uint32_t resolution = (uint32_t)ceil(exp2f(level * S) * H);
    const uint32_t resolution = (uint32_t)resolutions_list[level];
    //////////
    float ref_scale_rep = 1 / float(ref_scale);
    float ref_scale_rep_half = ref_scale_rep / 2;
    float ref_scale_rep_sq = ref_scale_rep * ref_scale_rep;

    float d0_start = inputs[0] - ref_scale_rep_half;
    d0_start = d0_start < 0? 0:d0_start;
    float d0_end = inputs[0] + ref_scale_rep_half;
    d0_end = d0_end > 1? 1:d0_end;
    float d1_start = inputs[1] - ref_scale_rep_half;
    d1_start = d1_start < 0? 0:d1_start;
    float d1_end = inputs[1] + ref_scale_rep_half;
    d1_end = d1_end > 1? 1:d1_end;

    uint32_t d0_start_pos = (uint32_t)ceilf(d0_start * float(resolution - 2) + 0.5);
    uint32_t d0_end_pos = (uint32_t)floorf(d0_end * float(resolution - 2) + 0.5);
    uint32_t d1_start_pos = (uint32_t)ceilf(d1_start * float(resolution - 2) + 0.5);
    uint32_t d1_end_pos = (uint32_t)floorf(d1_end * float(resolution - 2) + 0.5);

    scalar_t results[n_fearures] = {0}; // temp results in register
    float w_list[1 << 10] = {0};
    bool m_list[1 << 10] = {true};
    uint32_t zero_flag_list[1 << 10] = {0};
    uint32_t index_list[1 << 10] = {0};
    uint32_t pos_grid_local[2] = {0};
    float wn = 0.000000001;
    uint32_t idx = 0;

    float inputs0_ = inputs[0] * float(resolution - 2) + 0.5;
    float inputs1_ = inputs[1] * float(resolution - 2) + 0.5;

    #pragma unroll
    for (uint32_t d0 = d0_start_pos; d0 < d0_end_pos + 1; d0++) {
        if (abs(d0 - inputs0_) >= 4) continue;
        #pragma unroll
        for (uint32_t d1 = d1_start_pos; d1 < d1_end_pos + 1; d1++) {
            if (abs(d1 - inputs1_) >= 4) continue;

            pos_grid_local[0] = d0;
            pos_grid_local[1] = d1;

            uint32_t zero_flag = 0;
            #pragma unroll
            for (uint32_t d = 0; d < 2; d++) {
                if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                    zero_flag = 1;
                    break;
                }
            }

            bool m = true;
            if (true) {
                m = false;
                float scale_re = 1.0 / (float(resolution) - 2.0);
                uint32_t pos_g[2*2];  // xmin, ymin, zmin, xmax, ymax, zmax
                for (uint32_t d = 0; d < 2; d++) {
                    float points_n = (float(pos_grid_local[d]) - 0.5) * scale_re;

                    float pos_g1 = points_n - scale_re;
                    pos_g1 = pos_g1 * Rb;
                    pos_g1 = pos_g1 < 0? 0:pos_g1;
                    pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
                    pos_g[d] = int(pos_g1);

                    float pos_g2 = points_n + scale_re;
                    pos_g2 = pos_g2 * Rb;
                    pos_g2 = pos_g2 < 0? 0:pos_g2;
                    pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
                    pos_g[2 + d] = int(pos_g2);
                }

                for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
                        m = m | (binary_vxl[idx_a*Rb+idx_b]);
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }

            ///
            float d0_frac = (float(d0) - 0.5) / float(resolution - 2);
            float d1_frac = (float(d1) - 0.5) / float(resolution - 2);
            float dim0_abs = abs(d0_frac - inputs[0]);
            float dim1_abs = abs(d1_frac - inputs[1]);
            float w = ref_scale_rep_sq - ((dim0_abs + dim1_abs) * ref_scale_rep - dim0_abs * dim1_abs);
            ///

            w_list[idx] = w;
            m_list[idx] = m;
            zero_flag_list[idx] = zero_flag;
            if (zero_flag == 0 && m == true){
                uint32_t index = get_grid_index<2, n_fearures>(0, 0, hashmap_size, resolution, pos_grid_local);
                index_list[idx] = index;
                wn += w;
            }
            idx++;
        }
    }

    idx = 0;
    if (wn == 0) {
        wn += 1e-9;
    }
    float wn_re = 1.0 / wn;

    #pragma unroll
    for (uint32_t d0 = d0_start_pos; d0 < d0_end_pos + 1; d0++) {
        if (abs(d0 - inputs0_) >= 4) continue;
        #pragma unroll
        for (uint32_t d1 = d1_start_pos; d1 < d1_end_pos + 1; d1++) {
            if (abs(d1 - inputs1_) >= 4) continue;
            float www = w_list[idx] * wn_re;

            // note: m_list is designed for 3D grid context interpolation.
            // for normal point interpolation, m_list should always be all true.
            if (zero_flag_list[idx] == 0 && m_list[idx] == true) {  // valid point
                // writing to register (fast)
                #pragma unroll
                for (uint32_t ch = 0; ch < n_fearures; ch++) {
                    results[ch] += www * grid[index_list[idx] + ch];  // index is already multipled by n_fearures.
                }
            } else {
                for (uint32_t ch = 0; ch < n_fearures; ch++) {
                    results[ch] += 0;  // index is already multipled by n_fearures.
                }
            }
            idx++;
        }
    }
    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < n_fearures; ch++) {
        outputs[ch] = results[ch];
    }


}


template <typename scalar_t, uint32_t num_dim>
void avg_2D_forward_cuda(
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *outputs,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const bool *binary_vxl
    ) {
    // blocks and threads are defined here
    static constexpr uint32_t N_THREAD = 512;
    // div_round_up is (N + N_THREAD - 1) / N_THREAD
    const dim3 blocks_hashgrid = { div_round_up(N, N_THREAD), n_levels, 1 };
    switch (n_features) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: avg_2D_forward_kernel<scalar_t, 2, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 2: avg_2D_forward_kernel<scalar_t, 2, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 4: avg_2D_forward_kernel<scalar_t, 2, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 8: avg_2D_forward_kernel<scalar_t, 2, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 16: avg_2D_forward_kernel<scalar_t, 2, 16><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 32: avg_2D_forward_kernel<scalar_t, 2, 32><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels, Rb, ref_scale, binary_vxl); break;
        default: throw std::runtime_error{"GridEncoding: n_features must be 1, 2, 4, 8, 16 or 32."};
    }
}


void avg_2D_forward(
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor outputs,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const at::Tensor binary_vxl
    ) {

    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(outputs);
    CHECK_CUDA(binary_vxl);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(binary_vxl);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(outputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "avg_2D_forward",
    ([&] {
        avg_2D_forward_cuda<scalar_t, 2>(
            inputs.data_ptr<float>(),
            embeddings.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            outputs.data_ptr<scalar_t>(),
            N, n_features, n_levels, Rb, ref_scale,
            binary_vxl.data_ptr<bool>()
            );
    })
    );
}

// grad, inputs, embeddings, offsets_list, grad_embeddings, N, num_dim, n_fearures, n_levels, max_level, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation
template <typename scalar_t, uint32_t num_dim, uint32_t n_fearures, uint32_t n_features_per_thread>  // n_features_per_thread is n_features_per_thread
__global__ void avg_2D_backward_kernel(
    const scalar_t * __restrict__ grad,  // grad is the gradient from loss
    const float * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ grad_grid,   // same type as grad
    const uint32_t N, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const bool * __restrict__ binary_vxl
    ) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread / n_fearures;
    if (b >= N) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread - b * n_fearures;

    // locate
    grad_grid += offsets_list[level] * n_fearures;
    inputs += b * num_dim;
    grad += level * N * n_fearures + b * n_fearures + ch; // n_levels, N, n_fearures

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    scalar_t grad_cur[n_features_per_thread] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < n_features_per_thread; c++) {
        grad_cur[c] = grad[c];
    }

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    // const uint32_t resolution = (uint32_t)ceil(exp2f(level * S) * H);
    const uint32_t resolution = (uint32_t)resolutions_list[level];


    float ref_scale_rep = 1 / float(ref_scale);
    float ref_scale_rep_half = ref_scale_rep / 2;
    float ref_scale_rep_sq = ref_scale_rep * ref_scale_rep;

    float d0_start = inputs[0] - ref_scale_rep_half;
    d0_start = d0_start < 0? 0:d0_start;
    float d0_end = inputs[0] + ref_scale_rep_half;
    d0_end = d0_end > 1? 1:d0_end;
    float d1_start = inputs[1] - ref_scale_rep_half;
    d1_start = d1_start < 0? 0:d1_start;
    float d1_end = inputs[1] + ref_scale_rep_half;
    d1_end = d1_end > 1? 1:d1_end;

    uint32_t d0_start_pos = (uint32_t)ceilf(d0_start * float(resolution - 2) + 0.5);
    uint32_t d0_end_pos = (uint32_t)floorf(d0_end * float(resolution - 2) + 0.5);
    uint32_t d1_start_pos = (uint32_t)ceilf(d1_start * float(resolution - 2) + 0.5);
    uint32_t d1_end_pos = (uint32_t)floorf(d1_end * float(resolution - 2) + 0.5);

    scalar_t results[n_fearures] = {0}; // temp results in register
    float w_list[1 << 10] = {0};
    bool m_list[1 << 10] = {true};
    uint32_t zero_flag_list[1 << 10] = {0};
    uint32_t index_list[1 << 10] = {0};
    uint32_t pos_grid_local[2] = {0};
    float wn = 0.000000001;
    uint32_t idx = 0;

    float inputs0_ = inputs[0] * float(resolution - 2) + 0.5;
    float inputs1_ = inputs[1] * float(resolution - 2) + 0.5;

    #pragma unroll
    for (uint32_t d0 = d0_start_pos; d0 < d0_end_pos + 1; d0++) {
        if (abs(d0 - inputs0_) >= 4) continue;
        #pragma unroll
        for (uint32_t d1 = d1_start_pos; d1 < d1_end_pos + 1; d1++) {
            if (abs(d1 - inputs1_) >= 4) continue;
            pos_grid_local[0] = d0;
            pos_grid_local[1] = d1;

            uint32_t zero_flag = 0;
            #pragma unroll
            for (uint32_t d = 0; d < 2; d++) {
                if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                    zero_flag = 1;
                    break;
                }
            }

            bool m = true;
            if (true) {
                m = false;
                float scale_re = 1.0 / (float(resolution) - 2.0);
                uint32_t pos_g[2*2];  // xmin, ymin, zmin, xmax, ymax, zmax
                for (uint32_t d = 0; d < 2; d++) {
                    float points_n = (float(pos_grid_local[d]) - 0.5) * scale_re;

                    float pos_g1 = points_n - scale_re;
                    pos_g1 = pos_g1 * Rb;
                    pos_g1 = pos_g1 < 0? 0:pos_g1;
                    pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
                    pos_g[d] = int(pos_g1);

                    float pos_g2 = points_n + scale_re;
                    pos_g2 = pos_g2 * Rb;
                    pos_g2 = pos_g2 < 0? 0:pos_g2;
                    pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
                    pos_g[2 + d] = int(pos_g2);
                }

                for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
                        m = m | (binary_vxl[idx_a*Rb+idx_b]);
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }

            ///
            float d0_frac = (float(d0) - 0.5) / float(resolution - 2);
            float d1_frac = (float(d1) - 0.5) / float(resolution - 2);
            float dim0_abs = abs(d0_frac - inputs[0]);
            float dim1_abs = abs(d1_frac - inputs[1]);
            float w = ref_scale_rep_sq - ((dim0_abs + dim1_abs) * ref_scale_rep - dim0_abs * dim1_abs);
            ///

            w_list[idx] = w;
            m_list[idx] = m;
            zero_flag_list[idx] = zero_flag;
            if (zero_flag == 0 && m == true){
                uint32_t index = get_grid_index<2, n_fearures>(0, 0, hashmap_size, resolution, pos_grid_local);
                index_list[idx] = index;
                wn += w;
            }
            idx++;
        }
    }

    idx = 0;
    if (wn == 0) {
        wn += 1e-9;
    }
    float wn_re = 1.0 / wn;

    #pragma unroll
    for (uint32_t d0 = d0_start_pos; d0 < d0_end_pos + 1; d0++) {
        if (abs(d0 - inputs0_) >= 4) continue;
        #pragma unroll
        for (uint32_t d1 = d1_start_pos; d1 < d1_end_pos + 1; d1++) {
            if (abs(d1 - inputs1_) >= 4) continue;
            float www = w_list[idx] * wn_re;

            if (zero_flag_list[idx] == 0 && m_list[idx] == true) {

                // for atomicAdd, see https://www.cnblogs.com/biglucky/p/4283476.html. It is plus operation

                // atomicAdd for __half is slow (especially for large values), so we use __half2 if n_features_per_thread % 2 == 0
                // TODO: use float which is better than __half, if n_features_per_thread % 2 != 0
                if (std::is_same<scalar_t, at::Half>::value && n_features_per_thread % 2 == 0) {  // in this code it should be in this line
                    #pragma unroll
                    for (uint32_t c = 0; c < n_features_per_thread; c += 2) {
                        // process two __half at once (by interpreting as a __half2)
                        __half2 v = {(__half)(w_list[idx] * wn_re * grad_cur[c]), (__half)(www * grad_cur[c + 1])};
                        atomicAdd((__half2*)&grad_grid[index_list[idx] + c], v);
                    }
                // float, or __half when n_features_per_thread % 2 != 0 (which means n_fearures == 1)
                } else {
                    #pragma unroll
                    for (uint32_t c = 0; c < n_features_per_thread; c++) {
                        atomicAdd(&grad_grid[index_list[idx] + c], www * grad_cur[c]);
                    }
                }
            }
        }
        idx++;
    }
}

template <typename scalar_t, uint32_t num_dim>
void avg_2D_backward_cuda(
    const scalar_t *grad,
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *grad_embeddings,
    const uint32_t N, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const bool *binary_vxl
    ) {
    // blocks and threads are defined here
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t n_features_per_thread = std::min(2u, n_fearures); // n_features_per_thread
    const dim3 blocks_hashgrid = { div_round_up(N * n_fearures / n_features_per_thread, N_THREAD), n_levels, 1 };
    switch (n_fearures) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: avg_2D_backward_kernel<scalar_t, 2, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 2: avg_2D_backward_kernel<scalar_t, 2, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 4: avg_2D_backward_kernel<scalar_t, 2, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 8: avg_2D_backward_kernel<scalar_t, 2, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 16: avg_2D_backward_kernel<scalar_t, 2, 16, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, ref_scale, binary_vxl); break;
        case 32: avg_2D_backward_kernel<scalar_t, 2, 32, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, ref_scale, binary_vxl); break;
        default: throw std::runtime_error{"GridEncoding: n_fearures must be 1, 2, 4, 8, 16 or 32."};
    }
}

void avg_2D_backward(
    const at::Tensor grad,
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const at::Tensor binary_vxl
    ) {

    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(binary_vxl);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(binary_vxl);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(grad_embeddings);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "avg_2D_backward",
    ([&] {
        avg_2D_backward_cuda<scalar_t, 2>(
            grad.data_ptr<scalar_t>(),
            inputs.data_ptr<float>(),
            embeddings.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            grad_embeddings.data_ptr<scalar_t>(),
            N, n_features, n_levels, Rb, ref_scale,
            binary_vxl.data_ptr<bool>()
            );
    })
    );
}


/////////////////////////////


template <typename scalar_t, uint32_t num_dim, uint32_t n_features>
__global__ void cnt_np_embed_kernel(
    const short * __restrict__ inputs, // [N, 3]
    const scalar_t * __restrict__ grid,  // [520000, 4]
    scalar_t * __restrict__ outputs,  // [512, 512, 4, 2]
    const uint32_t N, const uint32_t resolution, const uint32_t hashmap_size, const uint32_t axis
    ) {

    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N
    if (b >= N) return;  // deprecate not needed threads
    inputs += b * 3;

    uint32_t pos_grid_local[3] = {inputs[0], inputs[1], inputs[2]};
    uint32_t index = get_grid_index<3, n_features>(0, 0, hashmap_size, resolution, pos_grid_local);
    uint32_t scale = resolution - 2;
    scalar_t results[n_features] = {0};
    #pragma unroll
    for (uint32_t ch = 0; ch < n_features; ch++) {
        results[ch] = grid[index + ch];  // index is already multipled by n_fearures.
    }
    uint32_t ot_loc = 0;

    for (uint32_t d=0; d < 3; d++){
        if (pos_grid_local[d] <= 0 || pos_grid_local[d] >= resolution - 1)  // 0 | 1, ...,  512 | 513
            return;
    }

    // now, pos_grid_local should be in [1, 2, 3, ..., 512]
    // now, pos_grid_local - 1 should be in [0, 1, 2, ..., 511]
    switch (axis) {
        case 0: ot_loc = (pos_grid_local[0] - 1) * scale * n_features * 2 + (pos_grid_local[1] - 1) * n_features * 2; break; // xy
        case 1: ot_loc = (pos_grid_local[0] - 1) * scale * n_features * 2 + (pos_grid_local[2] - 1) * n_features * 2; break; // xz
        case 2: ot_loc = (pos_grid_local[1] - 1) * scale * n_features * 2 + (pos_grid_local[2] - 1) * n_features * 2; break; // yz
    }

    for (uint32_t ch = 0; ch < n_features; ch++){
        if (results[ch] > 0.9) // 1
            // outputs[ot_loc + ch * 2 + 0] += 1;
            atomicAdd(&outputs[ot_loc + ch * 2 + 0], 1.0);
        else
            atomicAdd(&outputs[ot_loc + ch * 2 + 1], 1.0);
    }
}

template <typename scalar_t>
void cnt_np_embed_cuda(
    const short *inputs,
    const scalar_t *embeddings_clip,
    scalar_t *outputs,
    const uint32_t N, const uint32_t resolution, const uint32_t n_features, const uint32_t hashmap_size, const uint32_t axis
    ) {
    // blocks and threads are defined here
    const dim3 N_THREAD(256);  // 256
    const dim3 blocks_cnt((N+N_THREAD.x-1)/N_THREAD.x, 1, 1);
    switch (n_features) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: cnt_np_embed_kernel<scalar_t, 2, 1><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs, N, resolution, hashmap_size, axis); break;
        case 2: cnt_np_embed_kernel<scalar_t, 2, 2><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs, N, resolution, hashmap_size, axis); break;
        case 4: cnt_np_embed_kernel<scalar_t, 2, 4><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs, N, resolution, hashmap_size, axis); break;
        case 8: cnt_np_embed_kernel<scalar_t, 2, 8><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs, N, resolution, hashmap_size, axis); break;
        case 16: cnt_np_embed_kernel<scalar_t, 2, 16><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs, N, resolution, hashmap_size, axis); break;
        case 32: cnt_np_embed_kernel<scalar_t, 2, 32><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs, N, resolution, hashmap_size, axis); break;
        default: throw std::runtime_error{"GridEncoding: n_features must be 1, 2, 4, 8, 16 or 32."};
    }
}


void cnt_np_embed(
    const at::Tensor inputs, // [N, 3]
    const at::Tensor embeddings_clip,  // [520000, 4]
    at::Tensor outputs,  // [512, 512, 4, 2]
    const uint32_t N, const uint32_t resolution, const uint32_t n_features, const uint32_t hashmap_size, const uint32_t axis
    ) {

    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings_clip);
    CHECK_CUDA(outputs);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings_clip);
    CHECK_CONTIGUOUS(outputs);

    // CHECK_IS_INT(inputs);
    CHECK_IS_FLOATING(embeddings_clip);
    CHECK_IS_FLOATING(outputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings_clip.scalar_type(), "cnt_np_embed",
    ([&] {
        cnt_np_embed_cuda<scalar_t>(
            inputs.data_ptr<short>(),
            embeddings_clip.data_ptr<scalar_t>(),
            outputs.data_ptr<scalar_t>(),
            N, resolution, n_features, hashmap_size, axis
            );
    })
    );
}

template <typename scalar_t, uint32_t num_dim, uint32_t n_features>
__global__ void cnt_np_embed_backward_kernel(
    const short * __restrict__ inputs, // [N, 3]
    const scalar_t * __restrict__ grid,  // [520000, 4]
    const scalar_t * __restrict__ outputs_sum,  // [512, 512, 4, 1]
    const scalar_t * __restrict__ grad,  // [512, 512, 4, 2]
    scalar_t * __restrict__ grad_grid,  // [520000, 4]
    const uint32_t N, const uint32_t resolution, const uint32_t hashmap_size, const uint32_t axis
    ) {

    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N
    if (b >= N) return;  // deprecate not needed threads
    inputs += b * 3;
    float grad_v = 0;

    uint32_t pos_grid_local[3] = {inputs[0], inputs[1], inputs[2]};
    uint32_t index = get_grid_index<3, n_features>(0, 0, hashmap_size, resolution, pos_grid_local);
    uint32_t scale = resolution - 2;
    scalar_t results[n_features] = {0};
    #pragma unroll
    for (uint32_t ch = 0; ch < n_features; ch++) {
        results[ch] = grid[index + ch];  // index is already multipled by n_fearures.
    }
    uint32_t ot_loc = 0;

    for (uint32_t d=0; d < 3; d++){
        if (pos_grid_local[d] <= 0 || pos_grid_local[d] >= resolution - 1)  // 0 | 1, ...,  512 | 513
            return;
    }

    // now, pos_grid_local should be in [1, 2, 3, ..., 512]
    // now, pos_grid_local - 1 should be in [0, 1, 2, ..., 511]
    switch (axis) {
        case 0: ot_loc = (pos_grid_local[0] - 1) * scale * n_features * 2 + (pos_grid_local[1] - 1) * n_features * 2; break; // xy
        case 1: ot_loc = (pos_grid_local[0] - 1) * scale * n_features * 2 + (pos_grid_local[2] - 1) * n_features * 2; break; // xz
        case 2: ot_loc = (pos_grid_local[1] - 1) * scale * n_features * 2 + (pos_grid_local[2] - 1) * n_features * 2; break; // yz
    }
    uint32_t ot_loc_half = uint32_t(ot_loc / 2);

    for (uint32_t ch = 0; ch < n_features; ch++){
        grad_v = 1 / outputs_sum[ot_loc_half + ch * 1 + 0];
        if (results[ch] > 0.9) { // 1
            atomicAdd(&grad_grid[index + ch], grad_v * grad[ot_loc + ch * 2 + 0]);
        }
        else {
            atomicAdd(&grad_grid[index + ch], -grad_v * grad[ot_loc + ch * 2 + 1]);
        }
    }
}

template <typename scalar_t>
void cnt_np_embed_backward_cuda(
    const short *inputs,
    const scalar_t *embeddings_clip,
    const scalar_t *outputs_sum,
    const scalar_t *grad,
    scalar_t *grad_embeddings,
    const uint32_t N, const uint32_t resolution, const uint32_t n_features, const uint32_t hashmap_size, const uint32_t axis
    ) {
    // blocks and threads are defined here
    const dim3 N_THREAD(256);  // 256
    const dim3 blocks_cnt((N+N_THREAD.x-1)/N_THREAD.x, 1, 1);
    switch (n_features) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: cnt_np_embed_backward_kernel<scalar_t, 2, 1><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs_sum, grad, grad_embeddings, N, resolution, hashmap_size, axis); break;
        case 2: cnt_np_embed_backward_kernel<scalar_t, 2, 2><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs_sum, grad, grad_embeddings, N, resolution, hashmap_size, axis); break;
        case 4: cnt_np_embed_backward_kernel<scalar_t, 2, 4><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs_sum, grad, grad_embeddings, N, resolution, hashmap_size, axis); break;
        case 8: cnt_np_embed_backward_kernel<scalar_t, 2, 8><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs_sum, grad, grad_embeddings, N, resolution, hashmap_size, axis); break;
        case 16: cnt_np_embed_backward_kernel<scalar_t, 2, 16><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs_sum, grad, grad_embeddings, N, resolution, hashmap_size, axis); break;
        case 32: cnt_np_embed_backward_kernel<scalar_t, 2, 32><<<blocks_cnt, N_THREAD>>>(inputs, embeddings_clip, outputs_sum, grad, grad_embeddings, N, resolution, hashmap_size, axis); break;
        default: throw std::runtime_error{"GridEncoding: n_features must be 1, 2, 4, 8, 16 or 32."};
    }
}


void cnt_np_embed_backward(
    const at::Tensor inputs, // [N, 3]
    const at::Tensor embeddings_clip,  // [520000, 4]
    const at::Tensor outputs_sum,  // [512, 512, 4, 1]
    const at::Tensor grad,  // [512, 512, 4, 2]
    at::Tensor grad_embeddings,  // [520000, 4]
    const uint32_t N, const uint32_t resolution, const uint32_t n_features, const uint32_t hashmap_size, const uint32_t axis
    ) {

    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings_clip);
    CHECK_CUDA(outputs_sum);
    CHECK_CUDA(grad);
    CHECK_CUDA(grad_embeddings);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings_clip);
    CHECK_CONTIGUOUS(outputs_sum);
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(grad_embeddings);

    // CHECK_IS_INT(inputs);
    CHECK_IS_FLOATING(embeddings_clip);
    CHECK_IS_FLOATING(outputs_sum);
    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(grad_embeddings);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings_clip.scalar_type(), "cnt_np_embed_backward",
    ([&] {
        cnt_np_embed_backward_cuda<scalar_t>(
            inputs.data_ptr<short>(),
            embeddings_clip.data_ptr<scalar_t>(),
            outputs_sum.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            grad_embeddings.data_ptr<scalar_t>(),
            N, resolution, n_features, hashmap_size, axis
            );
    })
    );
}


////////////////////////////////////////////


// kernel_grid<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>
// (inputs, embeddings, offsets_list, outputs, N, n_levels, S, H, dy_dx, gridtype, align_corners, interp);
// N: N_rays
// n_levels: level
// S: log2(per_level_scale)
// H: base_resolution
////// One CPU kernel calls one GPU grid, one GPU grid contains several blocks, one block contains several threads
template <typename scalar_t, uint32_t num_dim, uint32_t n_fearures>  // <scalar_t, num_dim, 2 // num_dim: coords input_dim = 3
// __global__ means called by CPU and conducted by GPU
// always no return, so always void
__global__ void kernel_grid_mix2D(
    const float * __restrict__ inputs_xy, const float * __restrict__ inputs_xz, const float * __restrict__ inputs_yz,
    const scalar_t * __restrict__ grid_xy, const scalar_t * __restrict__ grid_xz, const scalar_t * __restrict__ grid_yz,
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ outputs,
    const uint32_t N, const uint32_t n_levels, const uint32_t Rb, const float PV,
    scalar_t * __restrict__ dy_dx,
    const bool * __restrict__ binary_vxl_2D_xy, const bool * __restrict__ binary_vxl_2D_xz, const bool * __restrict__ binary_vxl_2D_yz,
    const int * __restrict__ min_level_id,
    const uint32_t xy_len,
    const uint32_t xz_len,
    const uint32_t yz_len
    ) {
    // grid > block > thread
    // blockIdx.x is idx of the block along x axis, blockDim.x is th block width, threadIdx.x is idx along the width of thread
    // get the place of corresponding parallel computing point
    // get b: the index of [0, N_rays)
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N_rays axis

    if (b >= N) return;  // deprecate not needed threads

    const float * inputs = 0;
    const scalar_t * grid = 0;
    const bool * binary_vxl = 0;
    uint32_t b_prefix = 0;

    if (b >= 0 && b < xy_len) {
        inputs = inputs_xy;
        grid = grid_xy;
        binary_vxl = binary_vxl_2D_xy;
        b_prefix = 0;
    } else if (b >= xy_len && b < (xy_len + xz_len)) {
        inputs = inputs_xz;
        grid = grid_xz;
        binary_vxl = binary_vxl_2D_xz;
        b_prefix = xy_len;
    } else {
        inputs = inputs_yz;
        grid = grid_yz;
        binary_vxl = binary_vxl_2D_yz;
        b_prefix = xy_len + xz_len;
    }

    uint32_t level = 0;

    if (min_level_id) {
        level = min_level_id[b] + blockIdx.y;  // parallel along level axis, current level
        // printf(" A%dB%d", min_level_id[b], level);
    }
    else {
        level = blockIdx.y;
    }

    // locate  why these variables are changed? because they are pointers when defined? --> get the locate of the data to be processed
    grid += (uint32_t)offsets_list[level] * n_fearures;
    inputs += (b - b_prefix) * num_dim;
    outputs += blockIdx.y * N * n_fearures + b * n_fearures;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < n_fearures; ch++) {  // traverse each feature_dim
            outputs[ch] = 0;
        }
        if (dy_dx) {
            dy_dx += b * num_dim * n_levels * n_fearures + blockIdx.y * num_dim * n_fearures; // N n_levels num_dim n_fearures
            #pragma unroll
            for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
                #pragma unroll
                for (uint32_t ch = 0; ch < n_fearures; ch++) {  // traverse each feature_dim
                    dy_dx[d * n_fearures + ch] = 0;
                }
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    // exp2f(level * S) = 2 ^ (level*S) = 2 ^ (level*log2(per_level_scale)) = 2 ^ (log2(per_level_scale)*level) = per_level_scale ^ level
    // const uint32_t resolution = (uint32_t)ceil(exp2f(level * S) * H);
    const uint32_t resolution = (uint32_t)resolutions_list[level];

    // calculate coordinate (always use float for precision!)
    float pos[num_dim];
    float pos_deriv[num_dim];
    uint32_t pos_grid[num_dim];

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim

        pos[d] = inputs[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        pos_deriv[d] = 1.0f;
    }

    // interpolate
    scalar_t results[n_fearures] = {0}; // temp results in register
    float w_list[1 << num_dim] = {0};
    bool m_list[1 << num_dim] = {true};
    uint32_t zero_flag_list[1 << num_dim] = {0};
    uint32_t index_list[1 << num_dim] = {0};
    float wn = 0;

    #pragma unroll
    // idx = {0, 1, 2, 3, 4, 5, 6, 7}
    // here loop_num==8 is because there are 8 vertextes for interp 8=2**num_dim
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) { // why parallel is not applied for this loop?
        float w = 1;  // define weight for triblinear interp for different vertexes
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {  // (1 << d) = {1, 2, 4}
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }

        // gridtype: "hash" hashmap_size: hash map size of current level

        uint32_t zero_flag = 0;
        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                zero_flag = 1;
                break;
            }
        }

        bool m = true;
        if (binary_vxl){
            m = false;
            float scale_re = 1.0 / (float(resolution) - 2.0);
            uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
            #pragma unroll
            for (uint32_t d = 0; d < num_dim; d++) {
                float points_n = (float(pos_grid_local[d]) - 0.5) * scale_re;

                float pos_g1 = points_n - scale_re;
                pos_g1 = pos_g1 * Rb;
                pos_g1 = pos_g1 < 0? 0:pos_g1;
                pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
                pos_g[d] = int(pos_g1);

                float pos_g2 = points_n + scale_re;
                pos_g2 = pos_g2 * Rb;
                pos_g2 = pos_g2 < 0? 0:pos_g2;
                pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
                pos_g[num_dim + d] = int(pos_g2);
            }

            if (num_dim == 1) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[1]; idx_a++){
                    m = m | (binary_vxl[idx_a]);
                    if (m == true) break;
                }
            }
            if (num_dim == 2) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
                        m = m | (binary_vxl[idx_a*Rb+idx_b]);
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
            if (num_dim == 3) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[3]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[4]; idx_b++){
                        #pragma unroll
                        for (int idx_c=pos_g[2]; idx_c<=pos_g[5]; idx_c++){
                            m = m | (binary_vxl[idx_a*Rb*Rb+idx_b*Rb+idx_c]);
                            if (m == true) break;
                        }
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
        }

        w_list[idx] = w;
        m_list[idx] = m;
        zero_flag_list[idx] = zero_flag;
        if (zero_flag == 0 && m == true){
            uint32_t index = get_grid_index<num_dim, n_fearures>(0, 0, hashmap_size, resolution, pos_grid_local);
            index_list[idx] = index;
            wn += w;
        }
    }

    if (wn == 0) {
        wn += 1e-9;
    }
    float wn_re = 1.0 / wn;

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {
        // note: m_list is designed for 3D grid context interpolation.
        // for normal point interpolation, m_list should always be all true.
        if (zero_flag_list[idx] == 0 && m_list[idx] == true) {  // valid point
            // writing to register (fast)
            #pragma unroll
            for (uint32_t ch = 0; ch < n_fearures; ch++) {
                results[ch] += w_list[idx] * wn_re * grid[index_list[idx] + ch];  // index is already multipled by n_fearures.
            }
        }
    }

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < n_fearures; ch++) {
        outputs[ch] = results[ch];
    }

}



template <typename scalar_t, uint32_t num_dim>
void grid_encode_mix2D_forward_cuda(
    const float *inputs_xy, const float *inputs_xz, const float *inputs_yz,
    const scalar_t *embeddings_xy, const scalar_t *embeddings_xz, const scalar_t *embeddings_yz,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *outputs,
    const uint32_t N, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb, const float PV,
    scalar_t *dy_dx,
    const bool *binary_vxl_2D_xy,
    const bool *binary_vxl_2D_xz,
    const bool *binary_vxl_2D_yz,
    const int *min_level_id,
    const uint32_t xy_len, const uint32_t xz_len, const uint32_t yz_len
    ) {
    // blocks and threads are defined here
    static constexpr uint32_t N_THREAD = 512;
    // div_round_up is (N + N_THREAD - 1) / N_THREAD
    const dim3 blocks_hashgrid = { div_round_up(N, N_THREAD), n_levels, 1 };
    switch (n_fearures) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: kernel_grid_mix2D<scalar_t, 2, 1><<<blocks_hashgrid, N_THREAD>>>(inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len); break;
        case 2: kernel_grid_mix2D<scalar_t, 2, 2><<<blocks_hashgrid, N_THREAD>>>(inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len); break;
        case 4: kernel_grid_mix2D<scalar_t, 2, 4><<<blocks_hashgrid, N_THREAD>>>(inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len); break;
        case 8: kernel_grid_mix2D<scalar_t, 2, 8><<<blocks_hashgrid, N_THREAD>>>(inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len); break;
        case 16: kernel_grid_mix2D<scalar_t, 2, 16><<<blocks_hashgrid, N_THREAD>>>(inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len); break;
        case 32: kernel_grid_mix2D<scalar_t, 2, 32><<<blocks_hashgrid, N_THREAD>>>(inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, outputs, N, n_levels, Rb, PV, dy_dx, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len); break;
        default: throw std::runtime_error{"GridEncoding: n_fearures must be 1, 2, 4, 8, 16 or 32."};
    }

}

void grid_encode_mix2D_forward(
    const at::Tensor inputs_xy, const at::Tensor inputs_xz, const at::Tensor inputs_yz,
    const at::Tensor embeddings_xy, const at::Tensor embeddings_xz, const at::Tensor embeddings_yz,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor outputs,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb, const float PV,
    at::optional<at::Tensor> dy_dx,
    const at::optional<at::Tensor> binary_vxl_2D_xy, const at::optional<at::Tensor> binary_vxl_2D_xz, const at::optional<at::Tensor> binary_vxl_2D_yz,
    const at::optional<at::Tensor> min_level_id, const uint32_t xy_len, const uint32_t xz_len, const uint32_t yz_len
    ) {

    CHECK_CUDA(inputs_xy); CHECK_CUDA(inputs_xz); CHECK_CUDA(inputs_yz);
    CHECK_CUDA(embeddings_xy); CHECK_CUDA(embeddings_xz); CHECK_CUDA(embeddings_yz);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(outputs);
    // CHECK_CUDA(dy_dx);

    CHECK_CONTIGUOUS(inputs_xy); CHECK_CONTIGUOUS(inputs_xz); CHECK_CONTIGUOUS(inputs_yz);
    CHECK_CONTIGUOUS(embeddings_xy); CHECK_CONTIGUOUS(embeddings_xz); CHECK_CONTIGUOUS(embeddings_yz);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(outputs);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs_xy); CHECK_IS_FLOATING(inputs_xz); CHECK_IS_FLOATING(inputs_yz);
    CHECK_IS_FLOATING(embeddings_xy); CHECK_IS_FLOATING(embeddings_xz); CHECK_IS_FLOATING(embeddings_yz);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(outputs);
    // CHECK_IS_FLOATING(dy_dx);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF is the action to lunch a kernal
    // embeddings.scalar_type() indicates the type of data, to decide the type of lunched kernal
    // "grid_encode_forward" indicates the name in traceback, when reporting error...
    // grid_encode_forward_cuda name of function
    // process: first generate an output (already done), and fill data into it.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings_xy.scalar_type(), "grid_encode_mix2D_forward",
    ([&] {
        grid_encode_mix2D_forward_cuda<scalar_t, 2>(
            inputs_xy.data_ptr<float>(), inputs_xz.data_ptr<float>(), inputs_yz.data_ptr<float>(),
            embeddings_xy.data_ptr<scalar_t>(), embeddings_xz.data_ptr<scalar_t>(), embeddings_yz.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            outputs.data_ptr<scalar_t>(),
            N, n_fearures, n_levels, max_level, Rb, PV,
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr,
            binary_vxl_2D_xy.has_value() ? binary_vxl_2D_xy.value().data_ptr<bool>() : nullptr,
            binary_vxl_2D_xz.has_value() ? binary_vxl_2D_xz.value().data_ptr<bool>() : nullptr,
            binary_vxl_2D_yz.has_value() ? binary_vxl_2D_yz.value().data_ptr<bool>() : nullptr,
            min_level_id.has_value() ? min_level_id.value().data_ptr<int>() : nullptr,
            xy_len, xz_len, yz_len
            );
    })
    );
}

// grad, inputs, embeddings, offsets_list, grad_embeddings, N, num_dim, n_fearures, n_levels, max_level, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation
template <typename scalar_t, uint32_t num_dim, uint32_t n_fearures, uint32_t n_features_per_thread>  // n_features_per_thread is n_features_per_thread
__global__ void kernel_grid_mix2D_backward(
    const scalar_t * __restrict__ grad,  // grad is the gradient from loss
    const float * __restrict__ inputs_xy, const float * __restrict__ inputs_xz, const float * __restrict__ inputs_yz,
    const scalar_t * __restrict__ grid_xy, const scalar_t * __restrict__ grid_xz, const scalar_t * __restrict__ grid_yz,
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ grad_grid,   // same type as grad
    const uint32_t N, const uint32_t n_levels, const uint32_t Rb,
    const bool * __restrict__ binary_vxl_2D_xy, const bool * __restrict__ binary_vxl_2D_xz, const bool * __restrict__ binary_vxl_2D_yz,
    const int * __restrict__ min_level_id,
    const uint32_t xy_len, const uint32_t xz_len, const uint32_t yz_len,
    const uint32_t exy_len, const uint32_t exz_len, const uint32_t eyz_len
    ) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread / n_fearures;

    if (b >= N) return;

    const float * inputs = 0;
    const scalar_t * grid = 0;
    const bool * binary_vxl = 0;
    uint32_t b_prefix = 0;

    if (b >= 0 && b < xy_len) {
        inputs = inputs_xy;
        grid = grid_xy;
        binary_vxl = binary_vxl_2D_xy;
        b_prefix = 0;
        grad_grid += 0;
    } else if (b >= xy_len && b < (xy_len + xz_len)) {
        inputs = inputs_xz;
        grid = grid_xz;
        binary_vxl = binary_vxl_2D_xz;
        b_prefix = xy_len;
        grad_grid += exy_len * n_fearures;
    } else {
        inputs = inputs_yz;
        grid = grid_yz;
        binary_vxl = binary_vxl_2D_yz;
        b_prefix = xy_len + xz_len;
        grad_grid += (exy_len + exz_len) * n_fearures;
    }

    uint32_t level = 0;

    if (min_level_id) {
        level = min_level_id[b] + blockIdx.y;  // parallel along level axis, current level
    }
    else {
        level = blockIdx.y;
    }

    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread - b * n_fearures;

    // locate
    grad_grid += offsets_list[level] * n_fearures;
    inputs += (b - b_prefix) * num_dim;
    grad += blockIdx.y * N * n_fearures + b * n_fearures + ch; // n_levels, N, n_fearures

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    // const uint32_t resolution = (uint32_t)ceil(exp2f(level * S) * H);
    const uint32_t resolution = (uint32_t)resolutions_list[level];

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[num_dim];
    uint32_t pos_grid[num_dim];

    // same as forward process
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        pos[d] = inputs[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    scalar_t grad_cur[n_features_per_thread] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < n_features_per_thread; c++) {
        grad_cur[c] = grad[c];
    }

    float w_list[1 << num_dim] = {0};
    bool m_list[1 << num_dim] = {true};
    uint32_t zero_flag_list[1 << num_dim] = {0};
    uint32_t index_list[1 << num_dim] = {0};
    float wn = 0;

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {
        float w = 1;
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }

        uint32_t zero_flag = 0;
        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if (pos_grid_local[d]==0 || pos_grid_local[d]==resolution-1) {
                zero_flag = 1;
                break;
            }
        }

        bool m = true;
        if (binary_vxl){
            m = false;
            float scale_re = 1.0 / (float(resolution) - 2.0);
            uint32_t pos_g[num_dim*2];  // xmin, ymin, zmin, xmax, ymax, zmax
            #pragma unroll
            for (uint32_t d = 0; d < num_dim; d++) {
                float points_n = (float(pos_grid_local[d]) - 0.5) * scale_re;

                float pos_g1 = points_n - scale_re;
                pos_g1 = pos_g1 * Rb;
                pos_g1 = pos_g1 < 0? 0:pos_g1;
                pos_g1 = pos_g1 > Rb-1? Rb-1:pos_g1;
                pos_g[d] = int(pos_g1);

                float pos_g2 = points_n + scale_re;
                pos_g2 = pos_g2 * Rb;
                pos_g2 = pos_g2 < 0? 0:pos_g2;
                pos_g2 = pos_g2 > Rb-1? Rb-1:pos_g2;
                pos_g[num_dim + d] = int(pos_g2);
            }

            if (num_dim == 1) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[1]; idx_a++){
                    m = m | (binary_vxl[idx_a]);
                    if (m == true) break;
                }
            }
            if (num_dim == 2) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[2]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[3]; idx_b++){
                        m = m | (binary_vxl[idx_a*Rb+idx_b]);
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
            if (num_dim == 3) {
                #pragma unroll
                for (int idx_a=pos_g[0]; idx_a<=pos_g[3]; idx_a++){
                    #pragma unroll
                    for (int idx_b=pos_g[1]; idx_b<=pos_g[4]; idx_b++){
                        #pragma unroll
                        for (int idx_c=pos_g[2]; idx_c<=pos_g[5]; idx_c++){
                            m = m | (binary_vxl[idx_a*Rb*Rb+idx_b*Rb+idx_c]);
                            if (m == true) break;
                        }
                        if (m == true) break;
                    }
                    if (m == true) break;
                }
            }
        }

        w_list[idx] = w;
        m_list[idx] = m;
        zero_flag_list[idx] = zero_flag;
        if (zero_flag == 0 && m == true){
            uint32_t index = get_grid_index<num_dim, n_fearures>(0, ch, hashmap_size, resolution, pos_grid_local);
            index_list[idx] = index;
            wn += w;
        }
    }

    if (wn == 0) {
        wn += 1e-9;
    }
    float wn_re = 1.0 / wn;

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {
        if (zero_flag_list[idx] == 0 && m_list[idx] == true) {

            // for atomicAdd, see https://www.cnblogs.com/biglucky/p/4283476.html. It is plus operation

            // atomicAdd for __half is slow (especially for large values), so we use __half2 if n_features_per_thread % 2 == 0
            // TODO: use float which is better than __half, if n_features_per_thread % 2 != 0
            if (std::is_same<scalar_t, at::Half>::value && n_features_per_thread % 2 == 0) {  // in this code it should be in this line
                #pragma unroll
                for (uint32_t c = 0; c < n_features_per_thread; c += 2) {
                    // process two __half at once (by interpreting as a __half2)
                    __half2 v = {(__half)(w_list[idx] * wn_re * grad_cur[c]), (__half)(w_list[idx] * wn_re * grad_cur[c + 1])};
                    atomicAdd((__half2*)&grad_grid[index_list[idx] + c], v);
                }
            // float, or __half when n_features_per_thread % 2 != 0 (which means n_fearures == 1)
            } else {
                #pragma unroll
                for (uint32_t c = 0; c < n_features_per_thread; c++) {
                    atomicAdd(&grad_grid[index_list[idx] + c], w_list[idx] * wn_re * grad_cur[c]);
                }
            }
        }
    }
}

template <typename scalar_t, uint32_t num_dim>
void grid_encode_mix2D_backward_cuda(
    const scalar_t *grad,
    const float *inputs_xy, const float *inputs_xz, const float *inputs_yz,
    const scalar_t *embeddings_xy, const scalar_t *embeddings_xz, const scalar_t *embeddings_yz,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *grad_embeddings,
    const uint32_t N, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb,
    scalar_t *dy_dx,
    scalar_t *grad_inputs,
    const bool *binary_vxl_2D_xy,
    const bool *binary_vxl_2D_xz,
    const bool *binary_vxl_2D_yz,
    const int *min_level_id,
    const uint32_t xy_len, const uint32_t xz_len, const uint32_t yz_len,
    const uint32_t exy_len, const uint32_t exz_len, const uint32_t eyz_len
    ) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t n_features_per_thread = std::min(2u, n_fearures); // n_features_per_thread
    const dim3 blocks_hashgrid = { div_round_up(N * n_fearures / n_features_per_thread, N_THREAD), n_levels, 1 };

    switch (n_fearures) {
        case 1: kernel_grid_mix2D_backward<scalar_t, num_dim, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len, exy_len, exz_len, eyz_len); break;
        case 2: kernel_grid_mix2D_backward<scalar_t, num_dim, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len, exy_len, exz_len, eyz_len); break;
        case 4: kernel_grid_mix2D_backward<scalar_t, num_dim, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len, exy_len, exz_len, eyz_len); break;
        case 8: kernel_grid_mix2D_backward<scalar_t, num_dim, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len, exy_len, exz_len, eyz_len); break;
        case 16: kernel_grid_mix2D_backward<scalar_t, num_dim, 16, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len, exy_len, exz_len, eyz_len); break;
        case 32: kernel_grid_mix2D_backward<scalar_t, num_dim, 32, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs_xy, inputs_xz, inputs_yz, embeddings_xy, embeddings_xz, embeddings_yz, offsets_list, resolutions_list, grad_embeddings, N, n_levels, Rb, binary_vxl_2D_xy, binary_vxl_2D_xz, binary_vxl_2D_yz, min_level_id, xy_len, xz_len, yz_len, exy_len, exz_len, eyz_len); break;
        default: throw std::runtime_error{"GridEncoding: n_fearures must be 1, 2, 4, 8, 16 or 32."};
    }
}



void grid_encode_mix2D_backward(
    const at::Tensor grad,
    const at::Tensor inputs_xy, const at::Tensor inputs_xz, const at::Tensor inputs_yz,
    const at::Tensor embeddings_xy, const at::Tensor embeddings_xz, const at::Tensor embeddings_yz,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_fearures, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb,
    const at::optional<at::Tensor> dy_dx,
    at::optional<at::Tensor> grad_inputs,
    const at::optional<at::Tensor> binary_vxl_2D_xy, const at::optional<at::Tensor> binary_vxl_2D_xz, const at::optional<at::Tensor> binary_vxl_2D_yz,
    const at::optional<at::Tensor> min_level_id,
    const uint32_t xy_len, const uint32_t xz_len, const uint32_t yz_len,
    const uint32_t exy_len, const uint32_t exz_len, const uint32_t eyz_len
    ) {

    CHECK_CUDA(grad);
    CHECK_CUDA(inputs_xy); CHECK_CUDA(inputs_xz); CHECK_CUDA(inputs_yz);
    CHECK_CUDA(embeddings_xy); CHECK_CUDA(embeddings_xz); CHECK_CUDA(embeddings_yz);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(grad_embeddings);
    // CHECK_CUDA(dy_dx);
    // CHECK_CUDA(grad_inputs);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs_xy); CHECK_CONTIGUOUS(inputs_xz); CHECK_CONTIGUOUS(inputs_yz);
    CHECK_CONTIGUOUS(embeddings_xy); CHECK_CONTIGUOUS(embeddings_xz); CHECK_CONTIGUOUS(embeddings_yz);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(grad_embeddings);
    // CHECK_CONTIGUOUS(dy_dx);
    // CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs_xy); CHECK_IS_FLOATING(inputs_xz); CHECK_IS_FLOATING(inputs_yz);
    CHECK_IS_FLOATING(embeddings_xy); CHECK_IS_FLOATING(embeddings_xz); CHECK_IS_FLOATING(embeddings_yz);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(grad_embeddings);
    // CHECK_IS_FLOATING(dy_dx);
    // CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "grid_encode_mix2D_backward", ([&] {
        grid_encode_mix2D_backward_cuda<scalar_t, 2>(
            grad.data_ptr<scalar_t>(),
            inputs_xy.data_ptr<float>(), inputs_xz.data_ptr<float>(), inputs_yz.data_ptr<float>(),
            embeddings_xy.data_ptr<scalar_t>(), embeddings_xz.data_ptr<scalar_t>(), embeddings_yz.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            grad_embeddings.data_ptr<scalar_t>(),
            N, n_fearures, n_levels, max_level, Rb,
            dy_dx.has_value() ? dy_dx.value().data_ptr<scalar_t>() : nullptr,
            grad_inputs.has_value() ? grad_inputs.value().data_ptr<scalar_t>() : nullptr,
            binary_vxl_2D_xy.has_value() ? binary_vxl_2D_xy.value().data_ptr<bool>() : nullptr,
            binary_vxl_2D_xz.has_value() ? binary_vxl_2D_xz.value().data_ptr<bool>() : nullptr,
            binary_vxl_2D_yz.has_value() ? binary_vxl_2D_yz.value().data_ptr<bool>() : nullptr,
            min_level_id.has_value() ? min_level_id.value().data_ptr<int>() : nullptr,
            xy_len, xz_len, yz_len,
            exy_len, exz_len, eyz_len
            );
    }));

}
