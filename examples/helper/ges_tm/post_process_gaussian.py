import numpy as np
from plyfile import PlyData, PlyElement
import os


def dequantize_positions(q_coords,  meta_dir):
    def f_inv(value, bits):
        scale = 2 ** bits
        return value / scale

    loaded_data = np.load(meta_dir + '/meta.npz')
    print(f"Matadata:{loaded_data}")
    min_xyz = loaded_data["min_xyz"]
    max_xyz = loaded_data["max_xyz"]
    b_pos = loaded_data["bitwidth"]

    # scale_factor = 256
    scale_factor = max_xyz - min_xyz
    # 反量化
    normalized_coords = f_inv(q_coords, b_pos)  # 去量化
    coords_hat = normalized_coords * scale_factor + min_xyz  # 恢复到原始坐标范围

    def inverse_log_transform(y):
        return np.sign(y) * (np.expm1(np.abs(y)))

    coords_hat = inverse_log_transform(coords_hat)

    return coords_hat





def dequantize_opacity(op_q, b_op):
    def f_inv(value, bits):
        scale = 2 ** bits
        return value / scale

    # 反量化
    normalized_op = f_inv(op_q, b_op)  # 去量化到 [0, 1]
    op_hat = normalized_op * 25 - 7  # 恢复到原始范围
    return op_hat



def dequantize_scale(scale_q, b_s):
    def f_inv(value, bits):
        scale = 2 ** bits
        return value / scale

    # 反量化
    normalized_scale = f_inv(scale_q, b_s)  # 去量化到 [0, 1]
    scale_hat = normalized_scale * 30 - 26 # 恢复到原始范围
    return scale_hat


def dequantize_rot(rot_q, b_s):
    def f_inv(value, bits):
        scale = 2 ** bits
        return value / scale

    # 反量化
    normalized_rot = f_inv(rot_q, b_s)  # 去量化到 [0, 1]
    rot_hat = normalized_rot * 2 - 1  # 恢复到原始范围
    return rot_hat


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



def dequantize_sh(sh_q, b_s):
    def f_inv(value, bits):
        scale = 2 ** bits
        return value / scale
    # 反量化
    normalized_sh = f_inv(sh_q, b_s)  # 去量化到 [0, 1]
    # sh_hat = normalized_sh * (maxs-mins) + mins
    sh_hat = (normalized_sh-0.5)*8 # 恢复到原始范围
    return sh_hat


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


def inverse_load_ply(path, output_filename): # features_extra[:,:,level]
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    
    meta_dir = os.path.dirname(path)

    xyz = dequantize_positions(xyz, meta_dir)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]  # (n,1)
    opacities = dequantize_opacity(opacities, 10)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    features_dc[:, :, 0] = yuv_to_rgb(dequantize_sh(features_dc[:, :, 0], 10))
    f_dc = features_dc.reshape(features_dc.shape[0], -1)

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    for level in range(0, 15):  # 1-15
        features_extra[:, :, level] = yuv_to_rgb(dequantize_sh(features_extra[:, :, level], 10))
    f_rest = features_extra.reshape(features_extra.shape[0], -1)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = dequantize_scale(scales, 10)

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rots = dequantize_rot(rots, 10)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scales, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    PlyData([el]).write(output_filename)


if __name__ == "__main__":
    # G-PCC解码的的INRIA 3DGS ply模型 可修改
    gpcc_decoded_path = "output/bartender/point_cloud/iteration_30000/gpcc_decoded.ply"
    # 输出的meta数据文件 可修改
    output_min_xyz_path = "output/bartender/point_cloud/iteration_30000/min_xyz.npy"
    min_xyz = np.load(output_min_xyz_path)

    # 最终后处理的ply模型在 "output/bartender/point_cloud/iteration_30000/dequant_point_cloud.ply"路径下
    inverse_load_ply(gpcc_decoded_path, min_xyz) # 反量化
    

# 1. python pre_process_gaussian.py  # 仅一次 首先生成量化了的ply 按照m69429的预处理
# 2. 使用GeS-TM编解码
# ./tmc3 -c ./encoder_r04.cfg --uncompressedDataPath=bartender/point_cloud/iteration_30000/quant_point_cloud.ply --compressedStreamPath=bartender/point_cloud/iteration_30000/bartender.bin
# ./tmc3 -c ./decoder.cfg --compressedStreamPath=bartender/point_cloud/iteration_30000/bartender.bin --reconstructedDataPath=bartender/point_cloud/iteration_30000/gpcc_decoded.ply
# 3. python post_process_gaussian.py # 反量化+后处理 重建
# 4. python render.py -m output/bartender --skip_train --eval
# 5. python metrics.py -m output/bartender
