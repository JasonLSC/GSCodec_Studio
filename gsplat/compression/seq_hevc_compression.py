import json
import os
import subprocess
from dataclasses import dataclass, field, InitVar
import glob
import shutil
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from sympy import im
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from gsplat import compression
from gsplat.compression.outlier_filter import filter_splats
from gsplat.compression.sort import sort_splats
from gsplat.utils import inverse_log_transform, log_transform

@dataclass
class SeqHevcCompression:
    """Uses quantization and sorting to compress splats into mp4 files via libx265
      and uses K-means clustering to compress the spherical harmonic coefficents.

    .. warning::
        This class requires the `imageio <https://pypi.org/project/imageio/>`_,
        `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
        and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

    .. warning::
        This class might throw away a few lowest opacities splats if the number of
        splats is not a square number.

    .. note::
        The splats parameters are expected to be pre-activation values. It expects
        the following fields in the splats dictionary: "means", "scales", "quats",
        "opacities", "sh0", "shN". More fields can be added to the dictionary, but
        they will only be compressed using NPZ compression.

    References:
        - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
        - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.
    """

    use_sort: bool = True
    verbose: bool = True
    qp: Dict[str, Union[int, Dict[str, Any]]] = field(default_factory=lambda: {
        "means": -1,
        "opacities": 4,
        "quats": 4,
        "scales": 4,
        "sh0": 16,
        "shN":{
            "sh1": 20,
            "sh2": 24,
            "sh3": 28
        }
    })
    n_clusters: int = 32768
    debug: bool = False

    attribute_codec_registry: InitVar[Optional[Dict[str, str]]] = None

    compress_fn_map: Dict[str, Callable] = field(default_factory=lambda: {
        "means": _compress_video_hevc_16bit,
        "scales": _compress_video_hevc,
        "quats": _compress_quats_video_hevc,
        "opacities": _compress_video_hevc,
        "sh0": _compress_video_hevc,
        "shN": _compress_shN_video_hevc
        # "shN": _compress_masked_kmeans,
    })
    decompress_fn_map: Dict[str, Callable] = field(default_factory=lambda: {
        "means": _decompress_video_hevc_16bit,
        "scales": _decompress_video_hevc,
        "quats": _decompress_quats_video_hevc,
        "opacities": _decompress_video_hevc,
        "sh0": _decompress_video_hevc,
        "shN": _decompress_shN_video_hevc
        # "shN": _decompress_masked_kmeans,
    })

    def __post_init__(self, attribute_codec_registry):
        if attribute_codec_registry:
            available_functions = {
                "_compress_video_hevc_16bit": _compress_video_hevc_16bit,
                "_compress_video_hevc": _compress_video_hevc,
                "_compress_quats_video_hevc": _compress_quats_video_hevc,
                "_compress_shN_video_hevc": _compress_shN_video_hevc,
                # "_compress_masked_kmeans": _compress_masked_kmeans,

                "_decompress_video_hevc_16bit": _decompress_video_hevc_16bit,
                "_decompress_video_hevc": _decompress_video_hevc,
                "_decompress_quats_video_hevc": _decompress_quats_video_hevc,
                "_decompress_shN_video_hevc": _decompress_shN_video_hevc
                # "_decompress_masked_kmeans": _decompress_masked_kmeans,
            }

            for attr_name, attr_codec in attribute_codec_registry.items(): # go through the registry
                if attr_name in self.compress_fn_map and "encode" in attr_codec:
                    if attr_codec["encode"] in available_functions:
                        self.compress_fn_map[attr_name] = available_functions[attr_codec["encode"]]
                    else:
                        print(f"Warning: Unknown func: {attr_codec['encode']}")

                if attr_name in self.decompress_fn_map and "decode" in attr_codec:
                    if attr_codec["decode"] in available_functions:
                        self.decompress_fn_map[attr_name] = available_functions[attr_codec["decode"]]
                    else:
                        print(f"Warning: Unknown func: {attr_codec['decode']}")

    def _get_compress_fn(self, param_name: str) -> Callable:
        if param_name in self.compress_fn_map:
            return self.compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(self, param_name: str) -> Callable:
        if param_name in self.decompress_fn_map:
            return self.decompress_fn_map[param_name]
        else:
            return _decompress_npz
    
    def sort(self, splats_list: List[Dict]) -> Tensor:
        """Organize the list of splats into several sequences of attributs

        
        """
        first_splats = splats_list[0]

        n_gs = len(first_splats["means"])
        n_sidelen = int(np.ceil(n_gs**0.5))
        n_pad = n_sidelen**2 - n_gs
        if n_pad != 0:
            # splats = _crop_n_splats(splats, n_crop)
            first_splats = _pad_n_splats(first_splats, n_pad)
            print(
                f"Warning: Number of Gaussians was not square. Padded {n_pad} Gaussians."
            )
        
        _, sorted_indices = sort_splats(first_splats, return_indices=True, sort_with_shN=False)

        return sorted_indices
    
    def splats_list_to_attribute_seq(self, splats_list: List[Dict]) -> Dict[str, Tensor]:
        sample_splat = splats_list[0]
        attribute_names = list(sample_splat.keys())

        splats_sequences = {}
        for attr_name in attribute_names:
            attr_seq = [splat[attr_name] for splat in splats_list if attr_name in splat]
            attr_seq = torch.stack(attr_seq, dim=0)

            splats_sequences[attr_name] = attr_seq
        
        return splats_sequences
    
    def pad_attr_seq(self, splats_videos: Dict[str, Tensor]) -> Dict[str, Tensor]:
        n_gs = splats_videos["means"].size(1)
        n_sidelen = int(np.ceil(n_gs**0.5))
        n_pad = n_sidelen**2 - n_gs
        if n_pad != 0:
            print(
                f"Warning: Number of Gaussians was not square. Padded {n_pad} Gaussians."
            )
            for attr_name, splats_video in splats_videos.items():
                pad_shape = list(splats_video.shape)
                pad_shape[1] = n_pad
                if attr_name == "opacities":
                    pad_splats_video = -5 * torch.ones(pad_shape, dtype=splats_video.dtype, device=splats_video.device)
                elif attr_name == "scales":
                    pad_splats_video = -10 * torch.ones(pad_shape, dtype=splats_video.dtype, device=splats_video.device)
                else:
                    pad_splats_video = torch.zeros(pad_shape, dtype=splats_video.dtype, device=splats_video.device)

                splats_videos[attr_name] = torch.cat([splats_video, pad_splats_video], dim=1)
        
        return splats_videos
    
    def reorganize(self, splats_list: List[Dict]) -> Dict[str, Tensor]:
        # splat list to sequence of attributes
        seq_attr_dict = self.splats_list_to_attribute_seq(splats_list)
        # pad
        padded_splats_videos = self.pad_attr_seq(seq_attr_dict)

        if self.use_sort:
            ## get padded first splats
            ## sort and get indices
            sorted_indices = self.sort(splats_list)

            ## use indices to sort the sequences of attributes
            for attr_name, padded_splats_video in padded_splats_videos.items():
                padded_splats_videos[attr_name] = padded_splats_video[:, sorted_indices, ...]

        # reshape to 2d sequences
        n_gs = padded_splats_videos["means"].size(1)
        n_sidelen = int(n_gs**0.5)
        self.splats_videos = {}
        for attr_name, padded_splats_video in padded_splats_videos.items(): 
            ori_shape = list(padded_splats_video.shape)
            new_shape = [ori_shape[0]] + [n_sidelen, n_sidelen] + ori_shape[2:]
            self.splats_videos[attr_name] = padded_splats_video.reshape(new_shape)

            print(attr_name, padded_splats_video.shape)

        # proprocessing on splats_videos
        self.splats_videos["quats"] = F.normalize(self.splats_videos["quats"], dim=-1)

        return self.splats_videos

    def deorganize(self, splats_videos_c: Dict[str, Tensor]) -> List[Dict]:
        flattened_splats_videos = {}
        for attr_name, splats_video in splats_videos_c.items():
            ori_shape = list(splats_video.shape)
            new_shape = [ori_shape[0], ori_shape[1] * ori_shape[2]] + ori_shape[3:]

            flattened_splats_videos[attr_name] = splats_video.reshape(new_shape)

        n_frames = flattened_splats_videos["means"].size(0)
        splats_list = []

        for frame_idx in range(n_frames):
            splat_dict = {}
            for attr_name, attr_seq in flattened_splats_videos.items():
                splat_dict[attr_name] = attr_seq[frame_idx, ...]
            
            splats_list.append(splat_dict)
        
        return splats_list
    
    def compress(self, compress_dir: str) -> None:
        """Run compression

        Args:
            compress_dir (str): directory to save compressed files
        """

        # Param-specific preprocessing
        # splats["means"] = log_transform(splats["means"])
        self.splats_videos["quats"] = F.normalize(self.splats_videos["quats"], dim=-1)

        meta = {}
        for param_name in self.splats_videos.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                # "verbose": self.verbose,
                "n_sidelen": int(self.splats_videos["means"].size(1)),
                "qp": self.qp[param_name]
            }
            meta[param_name] = compress_fn(
                compress_dir, param_name, self.splats_videos[param_name], **kwargs
            )

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    def decompress(self, compress_dir: str) -> Dict[str, Tensor]:
        """Run decompression

        Args:
            compress_dir (str): directory that contains compressed files

        Returns:
            Dict[str, Tensor]: decompressed Gaussian splats
        """
        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        splats = {}
        for param_name, param_meta in meta.items():
            decompress_fn = self._get_decompress_fn(param_name)
            splats[param_name] = decompress_fn(compress_dir, param_name, param_meta)

        # Param-specific postprocessing
        # splats["means"] = inverse_log_transform(splats["means"])
        return splats

def _pad_n_splats(splats: Dict[str, Tensor], n_pad: int) -> Dict[str, Tensor]:
    for k, v in splats.items():
        pad_shape = list(v.shape)
        pad_shape[0] += n_pad
        padded_v = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        padded_v[:v.shape[0]] = v
        splats[k] = padded_v
    return splats

def _compress_video_hevc(
        compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, qp: int = 10, debug: bool = False
) -> Dict[str, Any]:
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    grid = params.reshape((n_frames, n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    video_norm = grid_norm.detach().cpu().numpy()

    video = (video_norm * (2**8 - 1)).round().astype(np.uint8)
    video = video.squeeze()
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), video)

    # save each frame
    for i in range(len(video)):
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}_frame{i:03d}.png"), video[i])
    
    # run ffmpeg libx265 to compress PNG file
    file_extension = ".265" if debug else ".mp4"
    video_file = os.path.join(compress_dir, f"{param_name}.{file_extension[1:]}")

    print(f"QP value of {param_name} is: {qp}")
    if param_name == "opacities":
        cmd = f"ffmpeg -i {compress_dir}/{param_name}_frame%03d.png -c:v libx265 -pix_fmt gray -x265-params \"qp={qp}\" " 
        cmd += f"{video_file}"
    else:
        cmd = f"ffmpeg -i {compress_dir}/{param_name}_frame%03d.png -c:v libx265 -x265-params \"qp={qp}\" "
        cmd += f"{video_file}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # remove png files
    png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_frame*.png")))
    for png_file in png_files:
        os.remove(png_file)
    
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta

def _decompress_video_hevc(compress_dir: str, param_name: str, meta: Dict[str, Any]):
    import imageio.v2 as imageio

    file_extension = meta["file_extension"]
    reader = imageio.get_reader(os.path.join(compress_dir, f"{param_name}.{file_extension[1:]}"), format='FFMPEG')

    frames = []
    for i, frame in enumerate(reader):
        frames.append(frame)
    
    video = np.stack(frames, axis=0)
    if param_name == "opacities":
        video = video[..., 0]
    
    # report the PSNR between reconstructed videos and original videos
    raw_video = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(255) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_video, video)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))

    video_norm = video / (2**8 - 1)

    grid_norm = torch.tensor(video_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_video_hevc_16bit(
        compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, qp: int = 10, debug: bool = False
) -> Dict[str, Any]:
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    grid = params.reshape((n_frames, n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    video_norm = grid_norm.detach().cpu().numpy()

    video = (video_norm * (2**16 - 1)).round().astype(np.uint16)
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), video,)

    video_l = video & 0xFF
    video_u = (video >> 8) & 0xFF

    # save each frame
    for i in range(len(video)):
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_l_frame{i:03d}.png"), video_l[i].astype(np.uint8)
        )
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_u_frame{i:03d}.png"), video_u[i].astype(np.uint8)
        )
    
    for byte_select in ['l', 'u']:
        file_extension = ".265" if debug else ".mp4"
        video_file = os.path.join(compress_dir, f"{param_name}_{byte_select}.{file_extension[1:]}")

        cmd = f"ffmpeg -i {compress_dir}/{param_name}_{byte_select}_frame%03d.png -c:v libx265 -x265-params \"lossless=1:preset=veryslow\" {video_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # remove png files
    if not debug:
        png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_*.png")))
        for png_file in png_files:
            os.remove(png_file)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta

def _decompress_video_hevc_16bit(
        compress_dir: str, param_name: str, meta: Dict[str, Any]
) -> Tensor:
    import imageio.v2 as imageio

    file_extension = meta["file_extension"]
    reader_l = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_l.{file_extension[1:]}"), format='FFMPEG')
    reader_u = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_u.{file_extension[1:]}"), format='FFMPEG')

    frames = []    
    for i, (frame_l, frame_u) in enumerate(zip(reader_l, reader_u)):
        frame_u = frame_u.astype(np.uint16)
        frame = (frame_u << 8) + frame_l
        frames.append(frame)
    
    video = np.stack(frames, axis=0)

    # report the PSNR between reconstructed videos and original videos
    raw_video = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(65535) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_video, video)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))

    video_norm = video / (2**16 - 1)

    grid_norm = torch.tensor(video_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params    

def _compress_quats_video_hevc(compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, qp: int = 10, debug: bool = False):
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    grid = params.reshape((n_frames, n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    video_norm = grid_norm.detach().cpu().numpy()

    video = (video_norm * (2**8 - 1)).round().astype(np.uint8)
    video = video.squeeze()
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), video,)

    video_w = video[..., 0] # [T, H, W]
    video_xyz = video[..., 1:] # [T, H, W, 3]

    for i in range(len(video)):
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}_w_frame{i:03d}.png"), video_w[i])
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}_xyz_frame{i:03d}.png"), video_xyz[i])

    # run ffmpeg libx265 to compress PNG file
    file_extension = ".265" if debug else ".mp4"

    video_file = os.path.join(compress_dir, f"{param_name}_w.{file_extension[1:]}")
    print(f"QP value of {param_name}_w is: {qp}")
    cmd = f"ffmpeg -i {compress_dir}/{param_name}_w_frame%03d.png -c:v libx265 -pix_fmt gray -x265-params \"qp={qp}\" {video_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    video_file = os.path.join(compress_dir, f"{param_name}_xyz.{file_extension[1:]}")
    print(f"QP value of {param_name}_xyz is: {qp}")
    cmd = f"ffmpeg -i {compress_dir}/{param_name}_xyz_frame%03d.png -c:v libx265 -x265-params \"qp={qp}\" {video_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # remove png files
    if not debug:
        png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_*.png")))
        for png_file in png_files:
            os.remove(png_file)
    
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta    

def _decompress_quats_video_hevc(
        compress_dir: str, param_name: str, meta: Dict[str, Any]
):
    import imageio.v2 as imageio

    file_extension = meta["file_extension"]
    reader_w = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_w.{file_extension[1:]}"), format='FFMPEG')
    reader_xyz = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_xyz.{file_extension[1:]}"), format='FFMPEG')

    frames = []
    for frame_w, frame_xyz in zip(reader_w, reader_xyz):
        frame = np.concatenate([frame_w[..., 0:1], frame_xyz], axis=-1)
        frames.append(frame)

    video = np.stack(frames, axis=0)

    # report the PSNR between reconstructed videos and original videos
    raw_video = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(255) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_video, video)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))   

    video_norm = video / (2**8 - 1)
    grid_norm = torch.tensor(video_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    # ori_params = torch.load(os.path.join(compress_dir, "quats.ckpt"))

    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params    

def _compress_shN_video_hevc(compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, qp: Dict[str, int], debug: bool = False):
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    shN_name_list = []
    for degree in range(1,4):
        for level in range(-degree, degree+1):
            shN_name_list.append(f"sh{degree}_{level}")

    grid = params # [T, H, W, 15, 3]
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    shN_norm = grid_norm.detach().cpu().numpy()

    shN_norm = (shN_norm * (2**8 - 1)).round().astype(np.uint8)
    shN_norm = shN_norm.squeeze()

    for f_id in range(n_frames):
        for shN_id, shN_name in enumerate(shN_name_list):
            image = shN_norm[f_id,:,:,shN_id,:]
            imageio.imwrite(os.path.join(compress_dir, f"{param_name}_{shN_name}_frame{f_id:03d}.png"), image)
    
    file_extension = ".265" if debug else ".mp4"
    for shN_id, shN_name in enumerate(shN_name_list):
        print(f"QP value of {shN_name} is: {qp[shN_name[0:3]]}")
        video_file = os.path.join(compress_dir, f"{param_name}_{shN_name}.{file_extension[1:]}")
        cmd = f"ffmpeg -i {compress_dir}/{param_name}_{shN_name}_frame%03d.png -c:v libx265 -x265-params \"qp={qp[shN_name[0:3]]}\" {video_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # remove png files
    if not debug:
        png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_*.png")))
        for png_file in png_files:
            os.remove(png_file)
    
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta

def _decompress_shN_video_hevc(
        compress_dir: str, param_name: str, meta: Dict[str, Any]
):
    import imageio.v2 as imageio

    shN_name_list = []
    for degree in range(1,4):
        for level in range(-degree, degree+1):
            shN_name_list.append(f"sh{degree}_{level}")

    file_extension = meta["file_extension"]

    shN_reader_list = []
    for shN_name in shN_name_list:
        shN_reader_list.append(imageio.get_reader(os.path.join(compress_dir, f"{param_name}_{shN_name}.{file_extension[1:]}"), format='FFMPEG'))

    shN_video_list = []
    for shN_reader in shN_reader_list: # loop on shN components
        shN_frames = []
        for shN_frame in shN_reader: # loop on frames
            shN_frames.append(shN_frame)
        shN_video = np.stack(shN_frames, axis=0)
        shN_video_list.append(shN_video)
    shN_videos = np.stack(shN_video_list, axis=3)

    shN_norm = shN_videos / (2**8 -1)

    grid_norm = torch.tensor(shN_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params 

def _compress_png(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 8-bit quantization and lossless PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_png(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters from PNG file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_hevc_kbit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, 
    quantization: int = 8, qp: int = 10, **kwargs
) -> Dict[str, Any]:
    """Compress scales parameters with k-bit quantization and HEVC compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**quantization - 1)).round().astype(np.uint8)
    img = img << (8 - quantization)
    img = img.squeeze()

    # save for check
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), img)

    png_file = os.path.join(compress_dir, f"{param_name}.png")
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    # run ffmpeg libx265 to compress PNG file
    mp4_file = os.path.join(compress_dir, f"{param_name}.mp4")
    # if param_name == "sh0":
    #     qp += 2
    print(f"QP value of {param_name} is: {qp}")
    if param_name == "opacities":
        cmd = f"ffmpeg -i {png_file} -c:v libx265 -pix_fmt gray -x265-params \"qp={qp}\" {mp4_file}"
    else:
        cmd = f"ffmpeg -i {png_file} -c:v libx265 -x265-params \"qp={qp}\" {mp4_file}"
    os.system(cmd)
    os.remove(png_file)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization, 
    }
    return meta


def _decompress_hevc_kbit(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters from mp4 file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    reader = imageio.get_reader(os.path.join(compress_dir, f"{param_name}.mp4"), format='FFMPEG')
    img = reader.get_data(0)
    reader.close()

    if param_name == "opacities":
        img = img[...,0]
        # import pdb; pdb.set_trace()

    # debug for check
    raw_img = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(255) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_img, img)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))
    # ---------------

    img = img >> (8 - meta["quantization"])
    img_norm = img / (2**meta["quantization"] - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_quats_hevc_kbit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, 
    quantization: int = 8, qp: int = 10, **kwargs
) -> Dict[str, Any]:
    """Compress scales parameters with k-bit quantization and HEVC compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**quantization - 1)).round().astype(np.uint8)
    img = img << (8 - quantization)
    img = img.squeeze()

    # save for check
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), img)

    # split 4 channels into 3+1 channels
    png_file_0 = os.path.join(compress_dir, f"{param_name}_u.png")
    imageio.imwrite(png_file_0, img[...,:3])

    png_file_1 = os.path.join(compress_dir, f"{param_name}_l.png")
    imageio.imwrite(png_file_1, img[...,3])

    # run ffmpeg libx265 to compress PNG file
    print(f"QP value of {param_name} is: {qp}")

    mp4_file_0 = os.path.join(compress_dir, f"{param_name}_u.mp4")
    mp4_file_1 = os.path.join(compress_dir, f"{param_name}_l.mp4")
    
    cmd = f"ffmpeg -i {png_file_0} -c:v libx265 -x265-params \"qp={qp}\" {mp4_file_0}"
    os.system(cmd)
    os.remove(png_file_0)

    cmd = f"ffmpeg -i {png_file_1} -c:v libx265 -pix_fmt gray -x265-params \"qp={qp}\" {mp4_file_1}"
    os.system(cmd)
    os.remove(png_file_1)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization, 
    }
    return meta


def _decompress_quats_hevc_kbit(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters from mp4 file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    reader = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_u.mp4"), format='FFMPEG')
    img_0 = reader.get_data(0)
    reader.close()

    reader = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_l.mp4"), format='FFMPEG')
    img_1 = reader.get_data(0)
    img_1 = img_1[...,0:1]
    reader.close()

    img = np.concatenate([img_0, img_1], axis=-1)

    # debug for check
    raw_img = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(255) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_img, img)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))
    # ---------------

    img = img >> (8 - meta["quantization"])
    img_norm = img / (2**meta["quantization"] - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_png_16bit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_l.png"), img_l.astype(np.uint8)
    )
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_u.png"), img_u.astype(np.uint8)
    )

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_png_16bit(
    compress_dir: str, param_name: str, meta: Dict[str, Any]
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
    img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
    img_u = img_u.astype(np.uint16)
    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_npz(
    compress_dir: str, param_name: str, params: Tensor, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with numpy's NPZ compression."""
    npz_dict = {"arr": params.detach().cpu().numpy()}
    save_fp = os.path.join(compress_dir, f"{param_name}.npz")
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    np.savez_compressed(save_fp, **npz_dict)
    meta = {
        "shape": params.shape,
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters with numpy's NPZ compression."""
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    params = torch.tensor(arr)
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_kmeans(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_clusters: int = 65536,
    quantization: int = 8,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run K-means clustering on parameters and save centroids and labels to a npz file.

    .. warning::
        TorchPQ must installed to use K-means clustering.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters to compress
        n_clusters (int): number of K-means clusters
        quantization (int): number of bits in quantization
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Any]: metadata
    """
    try:
        from torchpq.clustering import KMeans
    except:
        raise ImportError(
            "Please install torchpq with 'pip install torchpq' to use K-means clustering"
        )

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta
    # import pdb; pdb.set_trace()
    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)
    x = params.reshape(params.shape[0], -1).permute(1, 0).contiguous()
    labels = kmeans.fit(x)
    labels = labels.detach().cpu().numpy()
    centroids = kmeans.centroids.permute(1, 0)

    mins = torch.min(centroids)
    maxs = torch.max(centroids)
    centroids_norm = (centroids - mins) / (maxs - mins)
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = (
        (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)
    )
    labels = labels.astype(np.uint16)

    npz_dict = {
        "centroids": centroids_quant,
        "labels": labels,
    }
    np.savez_compressed(os.path.join(compress_dir, f"{param_name}.npz"), **npz_dict)
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
    }
    return meta


def _decompress_kmeans(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from K-means compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    npz_dict = np.load(os.path.join(compress_dir, f"{param_name}.npz"))
    centroids_quant = npz_dict["centroids"]
    labels = npz_dict["labels"].astype(np.int32) # uint16 -> int32

    centroids_norm = centroids_quant / (2 ** meta["quantization"] - 1)
    centroids_norm = torch.tensor(centroids_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    centroids = centroids_norm * (maxs - mins) + mins

    params = centroids[labels]
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_masked_kmeans(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_clusters: int = 4096, # 65536
    quantization: int = 8,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run K-means clustering on parameters and save centroids and labels to a npz file.

    .. warning::
        TorchPQ must installed to use K-means clustering.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters to compress
        n_clusters (int): number of K-means clusters
        quantization (int): number of bits in quantization
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Any]: metadata
        
    """
    try:
        from torchpq.clustering import KMeans
    except:
        raise ImportError(
            "Please install torchpq with 'pip install torchpq' to use K-means clustering"
        )

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta
    
    # get mask and save mask
    mask = (params > 0).any(dim=1).any(dim=1).reshape(-1)
    mask_flat = mask.cpu().numpy().astype(bool)
    n = len(mask_flat)
    n_bytes = (n + 7) // 8  # 需要的字节数
    bits = np.packbits(mask_flat)[:n_bytes]  # 打包成bytes
    bits.tofile(os.path.join(compress_dir, f"mask.bin"))

    # select vaild shN
    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)

    masked_params = params[mask]
    x = masked_params.reshape(masked_params.shape[0], -1).permute(1, 0).contiguous()

    labels = kmeans.fit(x)
    labels = labels.detach().cpu().numpy()
    centroids = kmeans.centroids.permute(1, 0)

    mins = torch.min(centroids)
    maxs = torch.max(centroids)
    centroids_norm = (centroids - mins) / (maxs - mins)
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = (
        (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)
    )
    labels = labels.astype(np.uint16)
    npz_dict = {
        "centroids": centroids_quant,
        "labels": labels,
    }
    np.savez_compressed(os.path.join(compress_dir, f"{param_name}.npz"), **npz_dict)
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
        "mask_bits": n,
        "mask_byte": n_bytes
    }
    return meta


def _decompress_masked_kmeans(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from K-means compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta
    
    # decode mask
    bits_loaded = np.fromfile(os.path.join(compress_dir, 'mask.bin'), dtype=np.uint8)
    mask_restored = np.unpackbits(bits_loaded)[:meta["mask_bits"]].astype(bool)
    mask = torch.from_numpy(mask_restored).reshape(meta["shape"][0])

    npz_dict = np.load(os.path.join(compress_dir, f"{param_name}.npz"))
    centroids_quant = npz_dict["centroids"]
    labels = npz_dict["labels"].astype(np.int32) # uint16 -> int32

    centroids_norm = centroids_quant / (2 ** meta["quantization"] - 1)
    centroids_norm = torch.tensor(centroids_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    centroids = centroids_norm * (maxs - mins) + mins

    params = centroids[labels]
    null_params = torch.zeros(meta["shape"], dtype=params.dtype) # null tensor
    null_params[mask] = params.reshape([params.shape[0]] + meta["shape"][1:])
    params = null_params
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def save_ply(splats: torch.nn.ParameterDict, path: str):
    from plyfile import PlyData, PlyElement

    means = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(means)
    sh0 = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    shN = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = splats["opacities"].detach().unsqueeze(1).cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()

    def construct_list_of_attributes(splats):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(splats["sh0"].shape[1]*splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(splats["shN"].shape[1]*splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats)]

    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, sh0, shN, opacities, scales, quats), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def save_params_into_ply_file(
    splats, path
):
    """Save parameters of Gaussian Splats into .ply file"""
    ply_dir = f"{path}/ply"
    os.makedirs(ply_dir, exist_ok=True)
    ply_file = ply_dir + "/pruned_splats.ply"
    save_ply(splats, ply_file)
    print(f"Saved parameters of splats into file: {ply_file}.")