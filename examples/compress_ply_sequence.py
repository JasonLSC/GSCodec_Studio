import json
import math
import os
import time
import shutil
import glob
from contextlib import nullcontext
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, ContextManager, TypedDict, Any

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, GSCDataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from gsplat import strategy
from gsplat.compression.entropy_coding_compression import EntropyCodingCompression
from gsplat.compression_simulation import simulation
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, load_ply
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import SeqHevcCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam
from gsplat.compression_simulation import CompressionSimulation
from gsplat.compression_simulation.entropy_model import Entropy_factorized_optimized_refactor, Entropy_gaussian

from helper.ges_tm.pre_process_gaussian import load_ply_and_quant
from helper.ges_tm.post_process_gaussian import inverse_load_ply


@dataclass
class CodecConfig:
    encode: str
    decode: str

@dataclass
class AttributeCodecs:
    means: CodecConfig = field(default_factory=lambda: CodecConfig("_compress_png_16bit", "_decompress_png_16bit"))
    scales: CodecConfig = field(default_factory=lambda: CodecConfig("_compress_factorized_ans", "_decompress_factorized_ans"))
    quats: CodecConfig = field(default_factory=lambda: CodecConfig("_compress_factorized_ans", "_decompress_factorized_ans"))
    opacities: CodecConfig = field(default_factory=lambda: CodecConfig("_compress_png", "_decompress_png"))
    sh0: CodecConfig = field(default_factory=lambda: CodecConfig("_compress_png", "_decompress_png"))
    shN: CodecConfig = field(default_factory=lambda: CodecConfig("_compress_masked_kmeans", "_decompress_masked_kmeans"))
    
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        return {
            attr: {"encode": getattr(self, attr).encode, "decode": getattr(self, attr).decode}
            for attr in ["means", "scales", "quats", "opacities", "sh0", "shN"]
        }

def default_qp_values() -> Dict[str, Union[int, Dict[str, Any]]]:
    """default qp values"""
    return {
        "means": -1,
        "opacities": 4,
        "quats": 4,
        "scales": 4,
        "sh0": 4,
        "shN":{
            "sh1": 4,
            "sh2": 4,
            "sh3": 4
        }
    }

@dataclass
class CompressionConfig:
    rate_point: str = "rp0"

@dataclass
class PCCompressionConfig(CompressionConfig):
    pcc_config_filename: str = "encoder_r05.cfg"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pcc_config_filename": self.pcc_config_filename
        }

@dataclass
class VideoCompressionConfig(CompressionConfig):
    # Use PLAS sort in compression or not
    use_sort: bool = True
    # Verbose or not
    verbose: bool = True
    # QP configuration - can be either int or dict for different attributes
    qp: Dict[str, Union[int, Dict[str, Any]]] = field(default_factory=default_qp_values)
    # Number of cluster of VQ for shN compression
    n_clusters: int = 32768
    # Maps attribute names to their codec functions
    attribute_codec_registry: Optional[AttributeCodecs] = field(default_factory=lambda: AttributeCodecs())
    # Enable All Intra coding mode
    use_all_intra: bool = False
    # Enable debug mode
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the CompressionConfig instance to a dictionary.
        If attribute_codec_registry is not None, it will be converted to a dictionary using its to_dict method.
        """
        # Get only attributes defined in VideoCompressionConfig
        video_compression_attrs = [
            "use_sort", "verbose", "qp", "n_clusters", 
            "use_all_intra", "debug"
        ]
        
        result = {attr: getattr(self, attr) for attr in video_compression_attrs}

        # handle attribute_codec_registry
        if self.attribute_codec_registry is not None:
            result["attribute_codec_registry"] = self.attribute_codec_registry.to_dict()

        return result

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["seq_hevc"]] = None
    # # Quantization parameters when set to hevc
    # qp: Optional[int] = None
    # Configuration for compression methods
    compression_cfg: CompressionConfig = field(
        default_factory=VideoCompressionConfig
    )

    # Enable profiler
    profiler_enabled: bool = False

    # Enable compression simulation
    compression_sim: bool = False
    # Name of quantization simulation strategy to use
    quantization_sim: Optional[Literal["round", "noise", "vq"]] = None

    # Enable entropy model
    entropy_model_opt: bool = False
    # Define the type of entropy model
    entropy_model_type: Literal["factorized_model", "gaussian_model"] = "factorized_model"
    # Bit-rate distortion trade-off parameter
    rd_lambda: float = 1e-2 # default: 1e-2
    # Steps to enable entropy model into training pipeline
    # factorized model:
    entropy_steps: Dict[str, int] = field(default_factory=lambda: {"means": -1, 
                                                                   "quats": 10_000, 
                                                                   "scales": 10_000, 
                                                                   "opacities": 10_000, 
                                                                   "sh0": 20_000, 
                                                                   "shN": 10_000})
    # gaussian model:
    # entropy_steps: Dict[str, int] = field(default_factory=lambda: {"means": -1, 
    #                                                                "quats": 10_000, 
    #                                                                "scales": 10_000, 
    #                                                                "opacities": 10_000, 
    #                                                                "sh0": 20_000, 
    #                                                                "shN": -1})

    # Enable shN adaptive mask
    shN_ada_mask_opt: bool = False
    # Steps to enable shN adaptive mask
    ada_mask_steps: int = 10_000
    # Strategy to obtain adaptive mask
    shN_ada_mask_strategy: Optional[str] = "learnable" # "gradient"
    
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    # data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Scene type
    scene_type: Literal["GSC", "default"] = "default"
    # Test view id
    test_view_id: Optional[List[int]] = None

    lpips_net: Literal["vgg", "alex"] = "alex"

    ### specific for I-3DGS compression
    # folder containing plys
    ply_dir: str = ""
    # folder containing colmap
    data_dir: str = ""
    # frame num
    frame_num: int = 1
    # anchor type
    anchor_type: Literal["video", "pcc"] = "video"

class Runner:
    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        # self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        # os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # frame num
        self.frame_num = cfg.frame_num

        # load ply sequences
        self.splats_list = self.load_ply_sequences(cfg.ply_dir, cfg.frame_num)

        # load dataset
        self.trainset_list, self.valset_list = self.set_up_datasets(cfg.data_dir, cfg.frame_num, cfg)

        self.compression_cfg = cfg.compression_cfg.to_dict()
        if cfg.compression == "seq_hevc":
            self.compression_method = SeqHevcCompression(**self.compression_cfg)

    def load_ply_sequences(
        self, ply_dir: str, frame_num: int
    ) -> List[torch.nn.ParameterDict]:
        self.ply_filename_list = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))

        splats_list = []
        for filename in tqdm.tqdm(self.ply_filename_list[:frame_num], desc="Loading .ply file"):
            splats = load_ply(filename)
            splats_list.append(splats.to("cuda"))
        
        return splats_list
    
    def set_up_datasets(
        self, data_dir: str, frame_num: int, cfg: Config
    ) -> Tuple[List[Dataset], List[Dataset]]:
        all_items = sorted(glob.glob(os.path.join(data_dir, "*")))
        folders = [item for item in all_items if os.path.isdir(item)]

        trainset_list = []
        valset_list = []
        for folder in tqdm.tqdm(folders[:frame_num], desc="Loading colmap results"):
            parser = Parser(
                data_dir=folder,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
            trainset = GSCDataset(
                parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
                test_view_ids=cfg.test_view_id,
            )
            valset = GSCDataset(
                parser, 
                split="val", 
                test_view_ids=cfg.test_view_id,)
            
            trainset_list.append(trainset)
            valset_list.append(valset)
        
        return trainset_list, valset_list

    def compress(self, ):
        """Entry for running video anchor compression."""
        print("Running video anchor compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression"

        if os.path.exists(compress_dir):
            shutil.rmtree(compress_dir)
        os.makedirs(compress_dir)

        splats_videos = self.compression_method.reorganize(self.splats_list)
        self.compression_method.compress(compress_dir)
        video_splats_c = self.compression_method.decompress(compress_dir)
        splats_list_c = self.compression_method.deorganize(video_splats_c)

        for splats, splats_c in zip(self.splats_list, splats_list_c):
            for k in splats.keys():
                splats[k].data = splats_c[k].to(self.device)

        self.eval(stage="compress")

    def pcc_compress(self, ):
        """Entry for running pc anchor compression."""
        print("Running pc anchor compression...")
        import subprocess

        compress_dir = f"{cfg.result_dir}/compression"
        intermediate_dir = f"{cfg.result_dir}/intermediate"
        log_dir = f"{cfg.result_dir}/log"
        rec_dir = f"{cfg.result_dir}/rec"

        if os.path.exists(compress_dir) or os.path.exists(intermediate_dir):
            shutil.rmtree(compress_dir)
            shutil.rmtree(intermediate_dir)
            shutil.rmtree(log_dir)
            # shutil.rmtree(rec_dir)
        os.makedirs(compress_dir, exist_ok=True)
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(rec_dir, exist_ok=True)

        for f_id, ply_file in enumerate(self.ply_filename_list[:self.frame_num]):
            # preprocess: fixed-point quantization
            temp_frame_dir = os.path.join(intermediate_dir, f"frame{f_id:03d}")
            os.makedirs(temp_frame_dir, exist_ok=True)
            load_ply_and_quant(ply_file, temp_frame_dir)

            # encode
            print(f"Encode frame{f_id:03d} via GeS-TM.")
            quant_ply_file = temp_frame_dir + f"/quant_splats.ply"
            encoded_bin_file = compress_dir + f"/frame{f_id:03d}.bin"
            encode_cmd = [
                './helper/ges_tm/tmc3',
                '-c',
                f"./helper/ges_tm/{self.compression_cfg['pcc_config_filename']}",
                f'--uncompressedDataPath={quant_ply_file}',
                f'--compressedStreamPath={encoded_bin_file}'
            ]

            encode_log_file = os.path.join(log_dir, f"frame{f_id:03d}_encode_log.txt")
            with open(encode_log_file, 'w') as log_file:
                result = subprocess.run(encode_cmd, 
                                    capture_output=True, 
                                    text=True, # output text rather than byte
                                    )
                log_file.write(result.stdout)
                log_file.write(result.stderr)
            
            # decode
            print(f"Decode frame{f_id:03d} via GeS-TM.")
            decoded_ply_file = temp_frame_dir + f"/decoded_quant_splats.ply"
            decode_cmd = [
                './helper/ges_tm/tmc3',
                '-c',
                './helper/ges_tm/decoder.cfg',
                f'--compressedStreamPath={encoded_bin_file}',
                f'--reconstructedDataPath={decoded_ply_file}'
            ]

            decode_log_file = os.path.join(log_dir, f"frame{f_id:03d}_decode_log.txt")
            with open(decode_log_file, 'w') as log_file:

                start_time = time.time()
                result = subprocess.run(decode_cmd, 
                                    capture_output=True, 
                                    text=True, # output text rather than byte
                                    )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                log_file.write(result.stdout)
                log_file.write(f"\nExecution time of decoding: {elapsed_time:.3f} seconds\n")

            print(f"Execution time of decoding: {elapsed_time:.3f} seconds")

            # postprocess
            output_filename = os.path.join(rec_dir, f"frame{f_id:03d}.ply")
            inverse_load_ply(decoded_ply_file, output_filename)    

        splats_list_c = self.load_ply_sequences(rec_dir, self.frame_num)

        for splats, splats_c in zip(self.splats_list, splats_list_c):
            for k in splats.keys():
                splats[k].data = splats_c[k].to(self.device)
        
        self.eval(stage="compress")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        splats: Optional[torch.nn.ParameterDict] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        if splats is not None:
            means = splats["means"] # [N, 3]
            quats = splats["quats"] # [N, 4]
            scales = torch.exp(splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(splats["opacities"])  # [N,]
            sh0, shN = splats["sh0"], splats["shN"]
        else:
            raise NotImplementedError(f"Should pass splats dict.")
    
        colors = torch.cat([sh0, shN], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info
    
    @torch.no_grad()
    def eval(self, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size        

        # dict to save metrics of each frame
        seq_stats = defaultdict(dict) 
        # loop on frame
        for f_id, (splats, val_dataset, train_dataset) in enumerate(zip(self.splats_list, self.valset_list, self.trainset_list)):
            valloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False, num_workers=1
            )
            ellipse_time = 0
            metrics = defaultdict(list)            
            # loop on view
            for v_id, data in enumerate(valloader):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                masks = data["mask"].to(device) if "mask" in data else None
                height, width = pixels.shape[1:3]
                splats = splats.to(device)

                torch.cuda.synchronize()
                tic = time.time()
                colors, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks,
                    splats=splats # must need
                )  # [1, H, W, 3]
                torch.cuda.synchronize()
                ellipse_time += time.time() - tic

                colors = torch.clamp(colors, 0.0, 1.0)
                canvas_list = [pixels, colors]

                if world_rank == 0:
                    # write images 
                    # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy() # side by side
                    canvas = canvas_list[1].squeeze(0).cpu().numpy() # signle image
                    canvas = (canvas * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_frame{f_id:03d}_testv{v_id:03d}.png",
                        canvas,
                    )

                    pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                    metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                    metrics["lpips"].append(self.lpips(colors_p, pixels_p))            
        
            if world_rank == 0:
                ellipse_time /= len(valloader)

                stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
                stats.update(
                    {
                        "ellipse_time": ellipse_time,
                        "num_GS": len(splats["means"]),
                    }
                )
                print(
                    f"Metrics on frame{f_id}:"
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
                # write into dict
                seq_stats[f"frame{f_id:03d}"] = stats

        # calculate average
        total_metrics = {k: 0 for k, v in stats.items()}
        frame_count = len(seq_stats)
        
        for frame, metrics in seq_stats.items():
            for metric, value in metrics.items():
                total_metrics[metric] += value
        
        avg_metrics = {metric: total/frame_count for metric, total in total_metrics.items()}

        seq_stats[f"average"] = avg_metrics
        print(
            f"Average Metrics:"
            f"PSNR: {avg_metrics['psnr']:.3f}, SSIM: {avg_metrics['ssim']:.4f}, LPIPS: {avg_metrics['lpips']:.3f} "
            f"Time: {avg_metrics['ellipse_time']:.3f}s/image "
            f"Number of GS: {avg_metrics['num_GS']}"
        )
        # save metrics
        with open(f"{self.stats_dir}/{stage}.json", "w") as f:
            json.dump(seq_stats, f, indent=4)        

    def eval_with_gsc_ctc_metrics(self, ):
        from helper.mpeg_gsc.gsc_metric import run_QMIV_metric
        from pathlib import Path
        height, width = self.valset_list[0][0]["image"].shape[0:2]
        resolution = f"{width}x{height}"

        # path to save QMIV log
        # if os.path.exists(f"{self.cfg.result_dir}/log"):
        #     shutil.rmtree(f"{self.cfg.result_dir}/log")
        os.makedirs(f"{self.cfg.result_dir}/log", exist_ok=True)

        gsc_metrics_across_test_views = defaultdict(dict)
        for test_view_id in range(len(self.cfg.test_view_id)):
            render_YUV_filename = Path(f"{self.cfg.result_dir}/renders/compress_testv{test_view_id:03d}.yuv")
            ref_YUV_filename = Path(f"{self.cfg.result_dir}/renders/val_testv{test_view_id:03d}.yuv")
            saved_log_file = Path(f"{self.cfg.result_dir}/log/QMIV_testv{test_view_id:03d}.txt")

            gsc_metrics = run_QMIV_metric(render_YUV_filename,
                                          ref_YUV_filename,
                                          resolution=resolution,
                                          saved_log_file=saved_log_file,
                                          pix_fmt="yuv420p")
            gsc_metrics_across_test_views[f"testv{test_view_id:03d}"] = gsc_metrics
        
        metric_names = gsc_metrics_across_test_views[f"testv{0:03d}"].keys()
        for metric in metric_names:
            total = sum(gsc_metrics_across_test_views[f"testv{i:03d}"][metric] 
                    for i in range(len(self.cfg.test_view_id)))
            gsc_metrics_across_test_views["average"][metric] = total / len(self.cfg.test_view_id)
        
        with open(os.path.join(self.cfg.result_dir, "stats", "gsc_metrics.json"), "w") as fp:
            json.dump(gsc_metrics_across_test_views, fp, indent=4)

    def summary(self,):
        import pandas as pd
        def format_size(size_bytes):
            """Convert byte size to readable format (KB, MB, GB, etc.)"""
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.2f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.2f} MB"
            else:
                return f"{size_bytes/(1024**3):.2f} GB"
            
        # rate summary
        directory_path = os.path.join(self.cfg.result_dir, "compression")
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist")
            return
        
        # Store file and size information
        file_sizes = {}
        total_size = 0
        
        # Get file sizes from directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                file_sizes[item] = size
                total_size += size

        # Get bitrate
        Byte_to_Mbps = lambda filesize, n_frame: filesize / (1024 ** 2) / n_frame * 8 * 30
        bitrate = Byte_to_Mbps(total_size, self.frame_num)

        # Calculate percentage
        percentages = {name: (size / total_size) * 100 for name, size in file_sizes.items()}
    
        # Create table data
        table_data = []
        for name, size in sorted(file_sizes.items(), key=lambda x: x[1], reverse=True):
            size_formatted = format_size(size)
            percentage = percentages[name]
            table_data.append([name, size_formatted, f"{percentage:.2f}%"])
        
        # Create pandas DataFrame for table
        df = pd.DataFrame(table_data, columns=["Filename", "Size", "Percentage"])
        csv_path = os.path.join(self.cfg.result_dir, "stats", "memory_breakdown.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV file saved to: {csv_path}")

        # distortion summary
        with open(os.path.join(self.cfg.result_dir, "stats", "compress.json"), "r") as fp:
            quality_metrics = json.load(fp)
            avg_quality_metrics = quality_metrics["average"]
        
        # save summary into a json file
        rd_summary = {key: value for key, value in avg_quality_metrics.items() if key != "ellipse_time"}
        rd_summary["bitrate"] = bitrate
        with open(os.path.join(self.cfg.result_dir, "summary.json"), "w") as fp:
            json.dump(rd_summary, fp, indent=4)

    def stack_render_img_to_vid(self):
        # remove existing video files
        for ext in ["*.mp4", "*.yuv"]:
            for file in glob.glob(os.path.join(self.cfg.result_dir, "renders", ext)):
                os.remove(file)

        for stage in ["compress", "val"]:
            for test_view_id in range(len(self.cfg.test_view_id)):
                # png sequence to mp4 for visualization
                cmd = (f'ffmpeg -framerate 30 -i "{self.cfg.result_dir}/renders/{stage}_frame%03d_testv{test_view_id:03d}.png" '
                    f'-c:v libx264 -pix_fmt yuv420p -crf 20 -preset medium '
                    f'-profile:v high -level 4.1 -movflags +faststart "{self.cfg.result_dir}/renders/{stage}_testv{test_view_id:03d}.mp4"')
                
                print(f"Running: {cmd}")
                os.system(cmd)
                print(f"Video created for {stage}, test view {test_view_id}")

                # png sequence to yuv for MPEG GSC metrics
                cmd = (f'ffmpeg -framerate 30 -i "{self.cfg.result_dir}/renders/{stage}_frame%03d_testv{test_view_id:03d}.png" '
                    f'-c:v rawvideo -pix_fmt yuv420p '
                    f'"{self.cfg.result_dir}/renders/{stage}_testv{test_view_id:03d}.yuv"')
                
                print(f"Running: {cmd}")
                os.system(cmd)
                print(f"YUV Video created for {stage}, test view {test_view_id}")

def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    runner = Runner(local_rank, world_rank, world_size, cfg)
    
    runner.eval()
    if cfg.anchor_type == "video":
        runner.compress()
    elif cfg.anchor_type == "pcc":
        runner.pcc_compress()
    else:
        raise NotImplementedError(f"{cfg.anchor_type} Anchor has not been implemented.")
    
    runner.stack_render_img_to_vid()
    runner.eval_with_gsc_ctc_metrics()
    runner.summary()

if __name__ == "__main__":
    configs = {
        "pcc_compression_debug":(
            "Use PCCompression.",
            Config(
                anchor_type="pcc",
                compression_cfg=PCCompressionConfig()
            )
        ),
        "pcc_compression_rp0":(
            "Use PCCompression.",
            Config(
                anchor_type="pcc",
                compression_cfg=PCCompressionConfig(
                    pcc_config_filename="encoder_r08.cfg"
                )
            )
        ),
        "pcc_compression_rp1":(
            "Use PCCompression.",
            Config(
                anchor_type="pcc",
                compression_cfg=PCCompressionConfig(
                    pcc_config_filename="encoder_r07.cfg"
                )
            )
        ),
        "pcc_compression_rp2":(
            "Use PCCompression.",
            Config(
                anchor_type="pcc",
                compression_cfg=PCCompressionConfig(
                    pcc_config_filename="encoder_r06.cfg"
                )
            )
        ),
        "pcc_compression_rp3":(
            "Use PCCompression.",
            Config(
                anchor_type="pcc",
                compression_cfg=PCCompressionConfig(
                    pcc_config_filename="encoder_r05.cfg"
                )
            )
        ),
        "x265_compression_debug":(
            "Use HevcCompression.",
            Config(
                compression="seq_hevc",
                compression_cfg=VideoCompressionConfig(n_clusters=8192)
            )
        ),
        "x265_compression_rp0": (
            "Use HevcCompression.",
            Config(
                compression="seq_hevc",
                compression_cfg=VideoCompressionConfig(
                    qp={
                        "means": -1,
                        "opacities": 4,
                        "quats": 4,
                        "scales": 4,
                        "sh0": 4,
                        "shN": {
                            "sh1": 4,
                            "sh2": 4,
                            "sh3": 4
                        }
                    }
                )
            )
        ),
        "x265_compression_rp1": (
            "Use HevcCompression.",
            Config(
                compression="seq_hevc",
                compression_cfg=VideoCompressionConfig(
                    qp={
                        "means": -1,
                        "opacities": 4,
                        "quats": 10,
                        "scales": 10,
                        "sh0": 4,
                        "shN": {
                            "sh1": 16,
                            "sh2": 22,
                            "sh3": 28
                        }
                    }
                )
            )
        ),
        "x265_compression_rp2": (
            "Use HevcCompression.",
            Config(
                compression="seq_hevc",
                compression_cfg=VideoCompressionConfig(
                    qp={
                        "means": -1,
                        "opacities": 10,
                        "quats": 16,
                        "scales": 16,
                        "sh0": 10,
                        "shN": {
                            "sh1": 22,
                            "sh2": 28,
                            "sh3": 34
                        }
                    }
                )
            )
        ),
        "x265_compression_rp3": (
            "Use HevcCompression.",
            Config(
                compression="seq_hevc",
                compression_cfg=VideoCompressionConfig(
                    qp={
                        "means": -1,
                        "opacities": 16,
                        "quats": 22,
                        "scales": 22,
                        "sh0": 16,
                        "shN": {
                            "sh1": 28,
                            "sh2": 34,
                            "sh3": 40
                        }
                    }
                )
            )
        ),
    }
        
    cfg = tyro.extras.overridable_config_cli(configs)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
