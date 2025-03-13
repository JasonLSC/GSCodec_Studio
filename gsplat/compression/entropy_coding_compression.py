import json
import os
from dataclasses import dataclass, field, InitVar
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from gsplat.compression.outlier_filter import filter_splats
from gsplat.compression.sort import sort_splats
from gsplat.utils import inverse_log_transform, log_transform
from gsplat.compression_simulation.ops import STE_binary
from gsplat.compression_simulation.entropy_model import Entropy_gaussian



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
    n_clusters: int = 32768

    attribute_codec_registry: InitVar[Optional[Dict[str, str]]] = None

    compress_fn_map: Dict[str, Callable] = field(default_factory=lambda: {
        "means": _compress_png_16bit,
        "scales": _compress_factorized_ans,
        "quats": _compress_factorized_ans,
        "opacities": _compress_png,
        "sh0": _compress_png,
        "shN": _compress_masked_kmeans,
    })
    decompress_fn_map: Dict[str, Callable] = field(default_factory=lambda: {
        "means": _decompress_png_16bit,
        "scales": _decompress_factorized_ans,
        "quats": _decompress_factorized_ans,
        "opacities": _decompress_png,
        "sh0": _decompress_png,
        "shN": _decompress_masked_kmeans,
    })

    def __post_init__(self, attribute_codec_registry):
        if attribute_codec_registry:
            available_functions = {
                "_compress_png_16bit": _compress_png_16bit,
                "_compress_png": _compress_png,
                "_compress_factorized_ans": _compress_factorized_ans,
                "_compress_gaussian_ans": _compress_gaussian_ans,
                "_compress_masked_kmeans": _compress_masked_kmeans,

                "_decompress_png_16bit": _decompress_png_16bit,
                "_decompress_png": _decompress_png,
                "_decompress_factorized_ans": _decompress_factorized_ans,
                "_decompress_gaussian_ans": _decompress_gaussian_ans,
                "_decompress_masked_kmeans": _decompress_masked_kmeans,
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
                decoded_means = self.get_decompressed_means(compress_dir, meta["means"])
                # kwargs.update({"decoded_means": inverse_log_transform(splats["means"])}) # means w/o quant
                kwargs.update({"decoded_means": decoded_means})

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
            if decompress_fn == _decompress_gaussian_ans:
                decoded_means = inverse_log_transform(splats["means"])
                splats[param_name] = decompress_fn(compress_dir, param_name, param_meta, decoded_means)
            else:
                splats[param_name] = decompress_fn(compress_dir, param_name, param_meta)

        # Param-specific postprocessing
        splats["means"] = inverse_log_transform(splats["means"])
        return splats
    
    def get_decompressed_means(self, compress_dir: str, meta_means: Dict) -> Tensor:
        # with open(os.path.join(compress_dir, "meta.json"), "r") as f:
        #     meta = json.load(f)
        
        decompress_fn = self._get_decompress_fn("means")
        decoded_means = decompress_fn(compress_dir, "means", meta_means)

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

def _get_likelihood(symbols: np.array, bitwidth: int=8) -> np.array:
    pass

def _categorical_ans_encode(symbols: np.array, probabilities: np.array, save_path:str):
    try:
        import constriction
    except:
        raise ImportError(
            "Please install constriction with 'pip install constriction' to use ANS"
    )

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
    print("Compressing:", param_name)

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

    # def get_quantized_value(mins, maxs, num_ch=3):
    #     tensor = torch.arange(0, 256).unsqueeze(1).repeat(1, num_ch).to(mins.device) # [256, 3]
    #     q_step = (maxs - mins)/(2**8 - 1)

    #     quantized_value = mins + tensor * q_step

    #     return quantized_value, q_step

    # q_value, q_step = get_quantized_value(mins, maxs)

    # entropy_model = kwargs["entropy_model"].to(q_value.device)
    # import pdb; pdb.set_trace()
    # # factorized 预测
    # prob = entropy_model.get_likelihood(q_value, q_step)
    # prob = prob.t() # [2**8, C] -> [C, 2**8]

    # 从这里开始
    if param_name == "sh0":
        params_norm = params_norm.squeeze(1)
    symbols = (params_norm * (2**8 - 1)).round().astype(np.int32)
    
    # 统计
    prob = _get_prob(symbols, 8) # [C, 2**8]
    if isinstance(prob, torch.Tensor):
        prob = prob.contiguous().detach().cpu().numpy()
    np.save(os.path.join(compress_dir, f"{param_name}_prob.npy"), prob)
    # prob should be like CDF
    # print(prob.shape, prob.dtype)
    print(symbols.shape, symbols.dtype)
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
    try:
        import constriction
    except:
        raise ImportError(
            "Please install constriction with 'pip install constriction' to use ANS"
        )

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

def to_bin_stream(signed_sym: torch.Tensor, filename: str):
    # float to -1/+1
    # signed_sym = STE_binary.apply(params)
    # -1/+1 to unsigned sym
    unsigned_sym = ((signed_sym + 1)//2).int().numpy()

    # compute the length of byte array
    n_bytes = (len(unsigned_sym) + 7) // 8
    packed = bytearray(n_bytes)

    # pack bits into binstream
    for i, bit in enumerate(unsigned_sym):
        if bit:
            packed[i // 8] |= (1 << (i % 8))
    
    # save file
    with open(filename, 'wb') as f:
        f.write(packed)

def save_gaussian_entropy_model(param_name: str, entropy_model: torch.nn.Module, compress_dir: str) -> Dict[str, Any]:
    # save entropy model (context model) - hash grid
    # entropy_model -> param_regressor -> hash_grid     -> encoding_xyz & encoding_xy & encoding_xz & encoding_yz -> params
    #                                  -> mlp_regressor

    # save hash grid
    encoding_params_dict = {
        "encoding_xyz": STE_binary.apply(entropy_model.param_regressor.hash_grid.encoding_xyz.params).detach().cpu(), # value: -1/+1
        "encoding_xy": STE_binary.apply(entropy_model.param_regressor.hash_grid.encoding_xy.params).detach().cpu(),
        "encoding_xz": STE_binary.apply(entropy_model.param_regressor.hash_grid.encoding_xz.params).detach().cpu(),
        "encoding_yz": STE_binary.apply(entropy_model.param_regressor.hash_grid.encoding_yz.params).detach().cpu()
    }

    meta = {}
    for enc_name, v in encoding_params_dict.items():
        meta[f"{enc_name}_shape"] = list(v.shape)
        to_bin_stream(v.view(-1), os.path.join(compress_dir, f"{param_name}_entropy_model_{enc_name}.bin"))

    # save mlp
    torch.save(entropy_model.param_regressor.mlp_regressor.state_dict(), 
               os.path.join(compress_dir, f"{param_name}_entropy_model_mlp.ckpt"))
    
    return meta

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
    try:
        import constriction
    except:
        raise ImportError(
            "Please install constriction with 'pip install constriction' to use ANS"
        )
    
    meta = save_gaussian_entropy_model(param_name, entropy_model, compress_dir)

    # quantization
    mins = torch.amin(params, dim=0)
    maxs = torch.amax(params, dim=0)

    params_norm = (params - mins) / (maxs - mins)
    params_norm = params_norm.detach().cpu().numpy()

    symbols = (params_norm * (2**8 - 1)).round().astype(np.int32)

    # Query to obtain distribution parameters
    entropy_model = entropy_model.to("cuda")
    miu, sigma = entropy_model.get_means_and_scales(decoded_means.to("cuda"))

    # Since our symbols are obtained via shift & scale transformations from the original parameters,
    # the probability distributions have changed accordingly. Therefore, the queried distribution parameters
    # also need to be transformed correspondingly.
    miu_norm = ((miu - mins) / (maxs - mins) * (2**8 - 1)).detach().cpu().numpy().astype(np.float64)  
    sigma_norm = (sigma / (maxs - mins) * (2**8 - 1)).detach().cpu().numpy().astype(np.float64)

    ### save intermediate variables for debug
    # tensor_to_be_saved = {
    #     "decoded_means": decoded_means,
    #     "miu": miu.detach().cpu(),
    #     "sigma": sigma.detach().cpu(),
    #     "mins": mins.detach().cpu(),
    #     "maxs": maxs.detach().cpu(),
    # }
    # torch.save(tensor_to_be_saved, os.path.join(compress_dir, f"{param_name}_debug.ckpt"))

    # flatten  into sequence
    message, means, stds = symbols.ravel(), miu_norm.ravel(), sigma_norm.ravel()

    # setup entropy coder
    model_family = constriction.stream.model.QuantizedGaussian(0, 255)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, model_family, means, stds)

    # compress symbols into binstream and save the binstream
    compressed = encoder.get_compressed()
    compressed.tofile(os.path.join(compress_dir, f"{param_name}.bin"))

    # Decode the message:
    compressed = np.fromfile(os.path.join(compress_dir, f"{param_name}.bin"), dtype=np.uint32)
    decoder = constriction.stream.stack.AnsCoder(compressed) # (we could also just reuse `encoder`.)
    reconstructed = decoder.decode(model_family, means, stds)

    assert np.all(reconstructed == message)

    meta.update({
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    })
    return meta

def from_bin_stream(filename: str, tensor_length: int) -> torch.Tensor:
    # read binary file
    with open(filename, 'rb') as f:
        byte_data = f.read()
    
    # set up tensor to be saved
    unsigned_sym = torch.zeros(tensor_length, dtype=torch.int)
    
    # unpack byte array
    for i in range(tensor_length):
        byte_index = i // 8
        bit_position = i % 8
        
        # Check if specific bit is 1
        if byte_index < len(byte_data) and (byte_data[byte_index] & (1 << bit_position)):
            unsigned_sym[i] = 1
    
    # Convert 0/1 to -1/+1
    signed_sym = unsigned_sym * 2 - 1
    signed_sym = signed_sym.to(dtype=torch.float32)
    
    return signed_sym

def _decompress_gaussian_ans(
    compress_dir: str, param_name: str, meta: Dict[str, Any], decoded_means: Tensor,
) -> Tensor:
    try:
        import constriction
    except:
        raise ImportError(
            "Please install constriction with 'pip install constriction' to use ANS"
        )
    # load entropy model
    entropy_model = Entropy_gaussian(channel=meta["shape"][-1])

    # load mlp of entropy model
    entropy_model.param_regressor.mlp_regressor.load_state_dict(torch.load(os.path.join(compress_dir, f"{param_name}_entropy_model_mlp.ckpt")))

    # load embedings of entropy model
    for k, v in meta.items():
        if "encoding" in k:
            enc_name = k[:-6]
            tensor_length = np.prod(v)
            decoded_encoding = from_bin_stream(os.path.join(compress_dir, f"{param_name}_entropy_model_{enc_name}.bin"), tensor_length)
            decoded_encoding = decoded_encoding.reshape(meta[f"{k}"])

            encoding_instance = getattr(entropy_model.param_regressor.hash_grid, enc_name)
            encoding_instance.params.data = decoded_encoding

    # 
    entropy_model = entropy_model.to("cuda")
    miu, sigma = entropy_model.get_means_and_scales(decoded_means.to("cuda"))

    mins = torch.tensor(meta["mins"]).to("cuda")
    maxs = torch.tensor(meta["maxs"]).to("cuda")
    miu_norm = ((miu - mins) / (maxs - mins) * (2**8 - 1)).detach().cpu().numpy().astype(np.float64)
    sigma_norm = (sigma / (maxs - mins) * (2**8 - 1)).detach().cpu().numpy().astype(np.float64)

    # flatten into sequence
    means, stds = miu_norm.ravel(), sigma_norm.ravel()

    ### load intermediate variables in encoding side to check if encoding and decoding match
    # debug_dict = torch.load(os.path.join(compress_dir, f"{param_name}_debug.ckpt"))

    # load .bin file of compressed symbols
    compressed = np.fromfile(os.path.join(compress_dir, f"{param_name}.bin"), dtype=np.uint32)
    
    # instantiate entropy coder
    decoder = constriction.stream.stack.AnsCoder(compressed)
    model_family = constriction.stream.model.QuantizedGaussian(0, 255)

    # decoding
    reconstructed = decoder.decode(model_family, means, stds)

    # recover the tensor
    params_norm = reconstructed / (2**8 - 1)
    params_norm = torch.tensor(params_norm).reshape(meta["shape"])
    params_norm = params_norm.to(dtype=getattr(torch, meta["dtype"]))

    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])

    params = params_norm * (maxs - mins) + mins

    return params

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