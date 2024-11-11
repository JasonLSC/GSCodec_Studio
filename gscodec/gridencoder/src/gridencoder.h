#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

// inputs: [N, num_dim], float, in [0, 1]
// embeddings: [offsets[-1], n_features], float
// offsets: [n_levels + 1], uint32_t
// outputs: [N, n_levels * n_features], float

void grid_encode_forward(
    const at::Tensor inputs, 
    const at::Tensor embeddings, 
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list, 
    at::Tensor outputs, 
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb, const float PV,
    at::optional<at::Tensor> dy_dx,
    const at::optional<at::Tensor> binary_vxl,
    const at::optional<at::Tensor> min_level_id
    );

void grid_encode_backward(
    const at::Tensor grad, 
    const at::Tensor inputs, 
    const at::Tensor embeddings, 
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings, 
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb,
    const at::optional<at::Tensor> dy_dx, 
    at::optional<at::Tensor> grad_inputs,
    const at::optional<at::Tensor> binary_vxl,
    const at::optional<at::Tensor> min_level_id
    );

void grid_encode_mix2D_forward(
    const at::Tensor inputs_xy, const at::Tensor inputs_xz, const at::Tensor inputs_yz,
    const at::Tensor embeddings_xy, const at::Tensor embeddings_xz, const at::Tensor embeddings_yz,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor outputs,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb, const float PV,
    at::optional<at::Tensor> dy_dx,
    const at::optional<at::Tensor> binary_vxl_2D_xy, const at::optional<at::Tensor> binary_vxl_2D_xz, const at::optional<at::Tensor> binary_vxl_2D_yz,
    const at::optional<at::Tensor> min_level_id, const uint32_t xy_len, const uint32_t xz_len, const uint32_t yz_len
    );

void grid_encode_mix2D_backward(
    const at::Tensor grad,
    const at::Tensor inputs_xy, const at::Tensor inputs_xz, const at::Tensor inputs_yz,
    const at::Tensor embeddings_xy, const at::Tensor embeddings_xz, const at::Tensor embeddings_yz,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels, const uint32_t max_level, const uint32_t Rb,
    const at::optional<at::Tensor> dy_dx,
    at::optional<at::Tensor> grad_inputs,
    const at::optional<at::Tensor> binary_vxl_2D_xy, const at::optional<at::Tensor> binary_vxl_2D_xz, const at::optional<at::Tensor> binary_vxl_2D_yz,
    const at::optional<at::Tensor> min_level_id,
    const uint32_t xy_len, const uint32_t xz_len, const uint32_t yz_len,
    const uint32_t exy_len, const uint32_t exz_len, const uint32_t eyz_len
    );

void avg_2D_forward(
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor outputs,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const at::Tensor binary_vxl
    );

void avg_2D_backward(
    const at::Tensor grad,
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels, const uint32_t Rb, const uint32_t ref_scale,
    const at::Tensor binary_vxl
    );

void cnt_np_embed(
    const at::Tensor inputs, // [N, 4*4*4, 3]
    const at::Tensor embeddings_clip,  // [520000, 4]
    at::Tensor outputs,  // [512, 512, 4, 2]
    const uint32_t N, const uint32_t resolution, const uint32_t n_features, const uint32_t hashmap_size, const uint32_t axis
    );

void cnt_np_embed_backward(
    const at::Tensor inputs, // [N, 4*4*4, 3]
    const at::Tensor embeddings_clip,  // [520000, 4]
    const at::Tensor outputs_sum,  // [512, 512, 4, 1]
    const at::Tensor grad,  // [512, 512, 4, 2]
    at::Tensor grad_embeddings,  // [520000, 4]
    const uint32_t N, const uint32_t resolution, const uint32_t n_features, const uint32_t hashmap_size, const uint32_t axis
    );

#endif