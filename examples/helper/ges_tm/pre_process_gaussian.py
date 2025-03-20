import numpy as np
from plyfile import PlyData, PlyElement
import os


def quantize_positions(coords, b_pos, meta_dir):
    def f(value, bits):
        scale = 2 ** bits
        value_scaled = value * scale
        return np.clip(np.round(value_scaled), 0, scale - 1).astype(int)
    
    def log_transform(x):
        return np.sign(x) * np.log1p(np.abs(x))
    
    coords = log_transform(coords)

    # 计算 x, y, z 的最小值
    mins = np.min(coords, axis=0)  # (3,)
    maxs = np.max(coords, axis=0)  # (3,)
    np.savez(meta_dir + '/meta.npz', min_xyz=mins, max_xyz=maxs, bitwidth=b_pos)
    print(f"Save metadata in file: {meta_dir + '/meta.npz'}")

    scale_factor = maxs - mins
    # scale_factor = 256
    # 量化
    normalized_coords = (coords - mins) / scale_factor  # 标准化
    q_coords = f(normalized_coords, b_pos)  # 量化
    return q_coords


def quantize_opacity(opacity, b_op):
    def f(value, bits):
        scale = 2 ** bits
        value_scaled = value * scale
        return np.clip(np.round(value_scaled), 0, scale - 1).astype(int)

    # mins = np.min(opacity, axis=0)  # (3,)
    # maxs = np.max(opacity, axis=0)  # (3,)
    # scale_factor = maxs - mins
    # 量化
    # normalized_op = (opacity - mins) / scale_factor
    normalized_op = (opacity + 7) / 25  # 标准化到 [0, 1]
    op_q = f(normalized_op, b_op)  # 量化
    return op_q


def quantize_scale(scale, b_s):
    def f(value, bits):
        scale = 2 ** bits
        value_scaled = value * scale
        return np.clip(np.round(value_scaled), 0, scale - 1).astype(int)

    # 量化
    normalized_scale = (scale + 26) / 30  # 标准化到 [0, 1]
    scale_q = f(normalized_scale, b_s)  # 量化
    return scale_q

def quantize_rot(rot, b_s):
    def f(value, bits):
        scale = 2 ** bits
        value_scaled = value * scale
        return np.clip(np.round(value_scaled), 0, scale - 1).astype(int)

    # 量化
    normalized_rot = (rot + 1) / 2  # 标准化到 [0, 1]
    rot_q = f(normalized_rot, b_s)  # 量化
    return rot_q


def rgb_to_yuv(rgb_sh):
    # 定义转换矩阵
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51498, -0.10001]
    ])
    yuv_sh = np.dot(rgb_sh, transform_matrix.T)
    return yuv_sh

def yuv_to_rgb(yuv_sh):
    # 定义逆变换矩阵
    inverse_transform_matrix = np.array([
        [1, 0, 1.13983],
        [1, -0.39465, -0.58060],
        [1, 2.03211, 0]
    ])
    # 进行矩阵乘法
    rgb_sh = np.dot(yuv_sh, inverse_transform_matrix.T)
    return rgb_sh

def quantize_sh(sh, b_s):
    def f(value, bits):
        scale = 2 ** bits
        value_scaled = value * scale
        return np.clip(np.round(value_scaled), 0, scale - 1).astype(int)

    # mins = np.min(sh, axis=0)
    # maxs = np.max(sh, axis=0)
    # 量化
    normalized_sh = sh/8+0.5  # 标准化到 [0, 1]
    # import pdb;pdb.set_trace()
    # normalized_sh = (sh-mins) / (maxs-mins)
    sh_q = f(normalized_sh, b_s)  # 量化
    return sh_q


def construct_list_of_attributes():
    l = ['x', 'y', 'z']  # 'nx', 'ny', 'nz'
    # All channels except the 3 DC
    for i in range(3*1):
        l.append('f_dc_{}'.format(i))
    for i in range(3*15):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l



def load_ply_and_quant(path, quant_ply_path):
    plydata = PlyData.read(path)

    dir_path = quant_ply_path

    # position
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)

    xyz = quantize_positions(xyz, 16, dir_path)

    # opacity
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis] # (n,1)
    opacities = quantize_opacity(opacities, 10)

    # sh0
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    features_dc[:,:,0] = quantize_sh(rgb_to_yuv(features_dc[:,:,0]), 10)

    f_dc = features_dc.reshape(features_dc.shape[0], -1)

    # shN
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (3 + 1) ** 2 - 3 # 45
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))
    # min_rest=[]
    # max_rest=[]
    for level in range(0, 15): # 1-15
        features_extra[:,:,level] = quantize_sh(rgb_to_yuv(features_extra[:,:,level]), 10)
        # min_rest.append(min_dc)
        # max_rest.append(max_dc)

    f_rest = features_extra.reshape(features_extra.shape[0], -1)

    # scales
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = quantize_scale(scales, 10)

    # quats
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rots = quantize_rot(rots, 10)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scales, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    quant_path = os.path.join(dir_path, "quant_splats.ply")

    PlyData([el]).write(quant_path)


if __name__ == "__main__":
    # 输入的INRIA 3DGS原始ply模型文件 可修改
    path = "output/bartender/point_cloud/iteration_30000/point_cloud.ply"
    # 输出的INRIA 3DGS量化的ply模型文件 可修改
    path_quant = "output/bartender/point_cloud/iteration_30000/quant_point_cloud.ply"
    # 输出的meta数据文件 可修改
    output_min_xyz_path = "output/bartender/point_cloud/iteration_30000/min_xyz.npy"

    min_xyz = load_ply_and_quant(path) # 量化
    np.save(output_min_xyz_path, min_xyz)


# 1. python pre_process_gaussian.py  # 仅一次 首先生成量化了的ply 按照m69429的预处理
# 2. 使用GeS-TM编解码
# ./tmc3 -c ./encoder_r04.cfg --uncompressedDataPath=bartender/point_cloud/iteration_30000/quant_point_cloud.ply --compressedStreamPath=bartender/point_cloud/iteration_30000/bartender.bin
# ./tmc3 -c ./decoder.cfg --compressedStreamPath=bartender/point_cloud/iteration_30000/bartender.bin --reconstructedDataPath=bartender/point_cloud/iteration_30000/gpcc_decoded.ply
# 3. python post_process_gaussian.py # 反量化+后处理 重建
# 4. python render.py -m output/bartender --skip_train --eval
# 5. python metrics.py -m output/bartender
