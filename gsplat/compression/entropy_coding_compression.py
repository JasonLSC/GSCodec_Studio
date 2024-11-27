import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from gsplat.compression.outlier_filter import filter_splats
from gsplat.compression.sort import sort_splats
from gsplat.utils import inverse_log_transform, log_transform

try:
    import constriction
except:
    raise ImportError(
        "Please install constriction with 'pip install constriction' to use ANS"
    )


@dataclass
class EntropyCodingCompression:
    """Uses quantization and sorting to compress splats into PNG files and uses
    K-means clustering to compress the spherical harmonic coefficents.

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

    def _get_compress_fn(self, param_name: str) -> Callable:
        compress_fn_map = {
            "means": _compress_png_16bit,
            "scales": _compress_gaussian_ans,
            "quats": _compress_png_kbit,
            "opacities": _compress_png,
            "sh0": _compress_gaussian_ans,
            "shN": _compress_masked_kmeans,
        }
        if param_name in compress_fn_map:
            return compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(self, param_name: str) -> Callable:
        decompress_fn_map = {
            "means": _decompress_png_16bit,
            "scales": _decompress_gaussian_ans,
            "quats": _compress_png_kbit,
            "opacities": _decompress_png,
            "sh0": _decompress_gaussian_ans,
            "shN": _decompress_masked_kmeans,
        }
        if param_name in decompress_fn_map:
            return decompress_fn_map[param_name]
        else:
            return _decompress_npz

    def compress(self, compress_dir: str, splats: Dict[str, Tensor], entropy_models: Dict[str, Module] = None) -> None:
        """Run compression

        Args:
            compress_dir (str): directory to save compressed files
            splats (Dict[str, Tensor]): Gaussian splats to compress
        """
        if entropy_models is None:
            raise ValueError("EntropyCodingCompression should require entropy_models")

        # Param-specific preprocessing
        splats["means"] = log_transform(splats["means"])
        splats["quats"] = F.normalize(splats["quats"], dim=-1)

        # Oulier filtering
        outlier_filtering = True
        if outlier_filtering:
            # import pdb; pdb.set_trace()
            vaild_mask, splats = filter_splats(splats)

        # Sorting
        n_gs = len(splats["means"])
        n_sidelen = int(n_gs**0.5)
        n_crop = n_gs - n_sidelen**2
        if n_crop != 0:
            splats = _crop_n_splats(splats, n_crop)
            print(
                f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
            )

        if self.use_sort:
            splats = sort_splats(splats)

        # Compress
        meta = {}
        for param_name in splats.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                "n_sidelen": n_sidelen,
                "verbose": self.verbose,
            }
            if param_name in entropy_models:
                kwargs.update({"entropy_model": entropy_models[param_name]})
                decoded_means = self.get_decompressed_means(compress_dir)
                kwargs.update({"decoded_means": inverse_log_transform(splats["means"])})

            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], **kwargs
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
        splats["means"] = inverse_log_transform(splats["means"])
        return splats
    
    def get_decompressed_means(self, compress_dir: str) -> Tensor:
        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        
        decompress_fn = self._get_decompress_fn("means")
        decoded_means = decompress_fn(compress_dir, "means", meta["means"])

        decoded_means = inverse_log_transform(decoded_means)
        return decoded_means


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    opacities = splats["opacities"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices]
    return splats

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

def _get_prob(symbols: np.array, bitwidth: int=8) -> np.array:
    prob_list = []
    for i in range(symbols.shape[-1]):
        counts = np.bincount(symbols[..., i].ravel(), minlength=2**bitwidth)

        prob = counts / counts.sum()
        prob = prob.astype(np.float32)

        prob_list.append(prob)

    stacked_prob_npy = np.stack(prob_list, axis=0)

    return stacked_prob_npy

def _categorical_ans_encode(symbols: np.array, probabilities: np.array, save_path:str):
    num_symbols = symbols.shape[-1]

    message_list = []
    probabilities_list = []
    model_list = []
    for i in range(symbols.shape[0]):
        message_list.append(symbols[i])
        probabilities_list.append(probabilities[i])
        model_list.append(constriction.stream.model.Categorical(probabilities[i], perfect=False))

    coder = constriction.stream.stack.AnsCoder()
    
    # encode: 反着编
    for i in range(symbols.shape[0]-1,-1,-1):
        coder.encode_reverse(message_list[i], model_list[i])


    compressed = coder.get_compressed()
    compressed.tofile(save_path)
    compressed = np.fromfile(save_path, dtype=np.uint32)

def _compress_factorized_ans(
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

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    mins = torch.amin(params, dim=0)
    maxs = torch.amax(params, dim=0)
    params_norm = (params - mins) / (maxs - mins)
    params_norm = params_norm.detach().cpu().numpy()

    symbols = (params_norm * (2**8 - 1)).round().astype(np.int32)
    prob = _get_prob(symbols, 8)
    np.save(os.path.join(compress_dir, f"{param_name}_prob.npy"), prob)
    _categorical_ans_encode(symbols.transpose(), prob, os.path.join(compress_dir, f"{param_name}.bin"))

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_factorized_ans(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
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

    compressed = np.fromfile(os.path.join(compress_dir, f"{param_name}.bin"), dtype=np.uint32)
    probabilities = np.load(os.path.join(compress_dir, f"{param_name}_prob.npy")).astype(np.float32)

    models = []
    for i in range(meta["shape"][1]):
        models.append(constriction.stream.model.Categorical(probabilities[i], perfect=False))
    ans = constriction.stream.stack.AnsCoder(compressed)

    symbols = []
    for i in range(meta["shape"][1]):
        symbol = ans.decode(models[i], meta["shape"][0])
        symbols.append(symbol)
    
    symbols = np.stack(symbols, axis=0)

    decoded_symbols = symbols.transpose()

    params_norm = decoded_symbols / (2**8 - 1)

    params_norm = torch.tensor(params_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    params = params_norm * (maxs - mins) + mins

    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_gaussian_ans(
    compress_dir: str, param_name: str, params: Tensor, entropy_model: Module, decoded_means: Tensor, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with ANS given Gaussian entropy model.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        entropy_model (Module): entropy model 

    Returns:
        Dict[str, Any]: metadata
    """

    mins = torch.amin(params, dim=0)
    maxs = torch.amax(params, dim=0)

    params_norm = (params - mins) / (maxs - mins)
    params_norm = params_norm.detach().cpu().numpy()

    symbols = (params_norm * (2**8 - 1)).round().astype(np.int32)

    entropy_model = entropy_model.to("cuda")

    miu, sigma = entropy_model.get_means_and_scales(decoded_means.to("cuda"))

    miu_norm = ((miu - mins) / (maxs - mins) * (2**8 - 1)).detach().cpu().numpy().astype(np.float64)
    sigma_norm = (sigma / (maxs - mins) * (2**8 - 1)).detach().cpu().numpy().astype(np.float64)

    # TODO: compress embedding

    # ANS
    message, means, stds = symbols.ravel(), miu_norm.ravel(), sigma_norm.ravel()

    model_family = constriction.stream.model.QuantizedGaussian(0, 255)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, model_family, means, stds)

    # Get and print the compressed representation:
    compressed = encoder.get_compressed()
    compressed.tofile(os.path.join(compress_dir, f"{param_name}.bin"))
    compressed = np.fromfile(os.path.join(compress_dir, f"{param_name}.bin"), dtype=np.uint32)

    # Decode the message:
    decoder = constriction.stream.stack.AnsCoder(compressed) # (we could also just reuse `encoder`.)
    reconstructed = decoder.decode(model_family, means, stds)
    # print(f"Reconstructed message: {reconstructed}")
    assert np.all(reconstructed == message)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta

def _decompress_gaussian_ans(
    compress_dir: str, param_name: str, meta: Dict[str, Any]
) -> Tensor:
    
    pass

def _compress_png_kbit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, quantization: int = 8, **kwargs
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

    img = (img_norm * (2**quantization - 1)).round().astype(np.uint8)
    img = img << (8 - quantization)
    img = img.squeeze()
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization, 
    }
    return meta


def _decompress_png_kbit(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
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
    n_clusters: int = 32768, # 65536
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