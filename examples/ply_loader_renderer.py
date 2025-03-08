import json
import math
import os
import time
import shutil
from contextlib import nullcontext
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, ContextManager, TypedDict

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
from gsplat import compression
from gsplat.compression.entropy_coding_compression import EntropyCodingCompression
from gsplat.compression_simulation import simulation
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression, HevcCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

from gsplat.compression_simulation import CompressionSimulation
from gsplat.compression_simulation.entropy_model import Entropy_factorized_optimized_refactor, Entropy_gaussian

class ProfilerConfig:
    def __init__(self):
        self.enabled = False
        self.activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        self.wait = 1 
        self.warmup = 2 
        self.active = 30_000 
        
        self.schedule = self._create_schedule()
        
        self.on_trace_ready = torch.profiler.tensorboard_trace_handler('./log/profiler')
        self.record_shapes = True
        self.profile_memory = True
        self.with_stack = True
    
    def _create_schedule(self):
        return torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
        )
    
    def update_schedule(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.schedule = self._create_schedule()

class CompressionConfig(TypedDict, total=False):
    # Use PLAS sort in compression or not
    use_sort: Optional[bool] = field(default=None)
    # Verbose or not
    verbose: Optional[bool] = field(default=None) 
    # QP value for video coding
    qp: int = 4
    # Number of cluster of VQ for shN compression
    n_clusters: int = 32768

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .ckpt files. If provide, it will skip training, load .ckpt file, and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Path to the .ply files. If provide, it will skip training, load .ply file, and run evaluation only.
    ply_path: Optional[str] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png", "entropy_coding", "hevc"]] = None
    # # Quantization parameters when set to hevc
    # qp: Optional[int] = None
    # Configuration for compression methods
    compression_cfg: CompressionConfig = field(
        default_factory=CompressionConfig
    )

    # Enable profiler
    profiler_enabled: bool = False

    # Enable compression simulation
    compression_sim: bool = False
    # # Name of quantization simulation strategy to use
    # quantization_sim: Optional[Literal["round", "noise", "vq"]] = None

    # # Enable entropy model
    # entropy_model_opt: bool = False
    # # Define the type of entropy model
    # entropy_model_type: Literal["factorized_model", "gaussian_model"] = "factorized_model"
    # # Bit-rate distortion trade-off parameter
    # rd_lambda: float = 1e-2 # default: 1e-2
    # # Steps to enable entropy model into training pipeline
    # # factorized model:
    # entropy_steps: Dict[str, int] = field(default_factory=lambda: {"means": -1, 
    #                                                                "quats": 10_000, 
    #                                                                "scales": 10_000, 
    #                                                                "opacities": 10_000, 
    #                                                                "sh0": 20_000, 
    #                                                                "shN": 10_000})
    # # gaussian model:
    # # entropy_steps: Dict[str, int] = field(default_factory=lambda: {"means": -1, 
    # #                                                                "quats": 10_000, 
    # #                                                                "scales": 10_000, 
    # #                                                                "opacities": 10_000, 
    # #                                                                "sh0": 20_000, 
    # #                                                                "shN": -1})

    # # Enable shN adaptive mask
    # shN_ada_mask_opt: bool = False
    # # Steps to enable shN adaptive mask
    # ada_mask_steps: int = 10_000
    # # Strategy to obtain adaptive mask
    # shN_ada_mask_strategy: Optional[str] = "learnable" # "gradient"
    
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
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

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

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

def load_ply(path: str) -> torch.nn.ParameterDict:
    from plyfile import PlyData
    import torch
    import numpy as np

    # Read PLY file
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # Get total number of vertices
    n_vertices = vertices.count

    # Extract basic attributes (positions)
    means = np.stack((vertices['x'], vertices['y'], vertices['z']), axis=1)
    
    # Calculate dimensions for sh0 and shN
    sh0_size = len([prop for prop in vertices.properties if prop.name.startswith('f_dc_')])
    shN_size = len([prop for prop in vertices.properties if prop.name.startswith('f_rest_')])
    
    # Extract sh0 data
    sh0_data = np.zeros((n_vertices, sh0_size))
    for i in range(sh0_size):
        sh0_data[:, i] = vertices[f'f_dc_{i}']
    
    # Extract shN data
    shN_data = np.zeros((n_vertices, shN_size))
    for i in range(shN_size):
        shN_data[:, i] = vertices[f'f_rest_{i}']
    
    # Extract opacity data
    opacities = vertices['opacity'].reshape(-1, 1)
    
    # Extract scales data
    scale_size = len([prop for prop in vertices.properties if prop.name.startswith('scale_')])
    scales = np.zeros((n_vertices, scale_size))
    for i in range(scale_size):
        scales[:, i] = vertices[f'scale_{i}']
    
    # Extract quaternion data
    quat_size = len([prop for prop in vertices.properties if prop.name.startswith('rot_')])
    quats = np.zeros((n_vertices, quat_size))
    for i in range(quat_size):
        quats[:, i] = vertices[f'rot_{i}']
    
    # Reshape sh0 and shN to original dimensions
    sh0_dim2 = 3  # Assume 3, adjust based on actual data
    sh0_dim1 = sh0_size // sh0_dim2
    shN_dim2 = 3  # Assume 3, adjust based on actual data
    shN_dim1 = shN_size // shN_dim2
    
    sh0_data = sh0_data.reshape(-1, sh0_dim2, sh0_dim1).transpose(0, 2, 1)
    shN_data = shN_data.reshape(-1, shN_dim2, shN_dim1).transpose(0, 2, 1)
    
    # Convert to torch tensors and create ParameterDict
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(means.astype(np.float32))),
        "sh0": torch.nn.Parameter(torch.from_numpy(sh0_data.astype(np.float32))),
        "shN": torch.nn.Parameter(torch.from_numpy(shN_data.astype(np.float32))),
        "opacities": torch.nn.Parameter(torch.from_numpy(opacities.astype(np.float32)).squeeze(1)),
        "scales": torch.nn.Parameter(torch.from_numpy(scales.astype(np.float32))),
        "quats": torch.nn.Parameter(torch.from_numpy(quats.astype(np.float32)))
    })
    
    return splats

class Runner:
    """Engine for training and testing."""

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
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        if cfg.scene_type == "GSC" and cfg.test_view_id is not None: # GSC mode
            self.trainset = GSCDataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
                test_view_ids=cfg.test_view_id,
            )
            self.valset = GSCDataset(
                self.parser, 
                split="val", 
                test_view_ids=cfg.test_view_id,)
        else: # default mode
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # # Densification Strategy
        # self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        # if isinstance(self.cfg.strategy, DefaultStrategy):
        #     self.strategy_state = self.cfg.strategy.initialize_state(
        #         scene_scale=self.scene_scale
        #     )
        # elif isinstance(self.cfg.strategy, MCMCStrategy):
        #     self.strategy_state = self.cfg.strategy.initialize_state()
        # else:
        #     assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            elif  cfg.compression == "entropy_coding":
                self.compression_method = EntropyCodingCompression()
            elif cfg.compression == "hevc":
                self.compression_method = HevcCompression(**cfg.compression_cfg) # compression_cfg=cfg.compression_cfg
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
        
        # Profiler
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiler_config = ProfilerConfig()
        if cfg.profiler_enabled:
            self.profiler_config.enabled = True

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

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def get_profiler(self, tb_writer) -> ContextManager:
        if self.profiler_config.enabled:
            return torch.profiler.profile(
                activities=self.profiler_config.activities,
                schedule=self.profiler_config.schedule,
                # on_trace_ready=self.profiler_config.on_trace_ready, 
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_writer.log_dir),
                record_shapes=self.profiler_config.record_shapes,
                profile_memory=self.profiler_config.profile_memory,
                with_stack=self.profiler_config.with_stack
            )
        return nullcontext()

    def step_profiler(self):
        """step profiler"""
        if self.profiler is not None:
            self.profiler.step()

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        sh0, shN = self.splats["sh0"], self.splats["shN"]
        if self.cfg.compression_sim:
            means = self.comp_sim_splats["means"]  # [N, 3]
            quats = self.comp_sim_splats["quats"]  # [N, 4]
            scales = torch.exp(self.comp_sim_splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.comp_sim_splats["opacities"])  # [N,]
            sh0, shN = self.comp_sim_splats["sh0"], self.comp_sim_splats["shN"]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
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
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

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
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int, stage: str = "val"):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        num_imgs = len(self.parser.camtoworlds)

        camtoworlds_all = self.parser.camtoworlds[: num_imgs//2]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 6 #1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/{stage}_traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            # canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = canvas_list[0].squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/{stage}_traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"

        if os.path.exists(compress_dir):
            shutil.rmtree(compress_dir)
        os.makedirs(compress_dir)

        self.run_param_distribution_vis(self.splats, save_dir=f"{cfg.result_dir}/visualization/raw")
        
        if isinstance(self.compression_method, PngCompression):
            self.compression_method.compress(compress_dir, self.splats)
        elif isinstance(self.compression_method, EntropyCodingCompression):
            self.compression_method.compress(compress_dir, self.splats, self.entropy_models)
        elif isinstance(self.compression_method, HevcCompression):
            self.compression_method.compress(compress_dir, self.splats)
        else:
            raise NotImplementedError(f"The compression method is not implemented yet.")

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        
        self.run_param_distribution_vis(splats_c, save_dir=f"{cfg.result_dir}/visualization/quant")

        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")
        self.render_traj(step=step, stage="compress")

    @torch.no_grad()
    def run_param_distribution_vis(self, param_dict: Dict[str, Tensor], save_dir: str):
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)
        for param_name, value in param_dict.items():
            
            tensor_np = value.flatten().detach().cpu().numpy()
            min_val, max_val = tensor_np.min(), tensor_np.max()

            plt.figure(figsize=(6, 4))
            n, bins, patches = plt.hist(tensor_np, bins=50, density=False, alpha=0.7, color='b')

            for count, bin_edge in zip(n, bins):
                plt.text(bin_edge, count, f'{int(count)}', fontsize=8, va='bottom', ha='center')

            plt.annotate(f'Min: {min_val:.2f}', xy=(min_val, 0), xytext=(min_val, max(n) * 0.1),
                        arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green')

            plt.annotate(f'Max: {max_val:.2f}', xy=(max_val, 0), xytext=(max_val, max(n) * 0.1),
                        arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')

            plt.title(f'{param_name} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Density')

            plt.savefig(os.path.join(save_dir, f'{param_name}.png'))

            plt.close()
        
        print(f"Histograms saved in '{save_dir}' directory.")
    
    def load_entropy_model_from_ckpt(self, ckpt: Dict, entropy_model_type: str):
        self.entropy_models = {}
        for name, value in ckpt.items():
            if "_entropy_model" in name:
                attr_name = name[:(len(name) - len("_entropy_model"))]
                num_ch = ckpt["splats"][attr_name].shape[-1]
                if entropy_model_type == "factorized_model":
                    # TODO
                    if attr_name == "scales" or attr_name == "sh0":
                        filters = (3, 3)
                    else:
                        filters = (3, 3, 3)
                    entropy_model = Entropy_factorized_optimized_refactor(channel=num_ch, filters=filters)

                elif entropy_model_type == "gaussian_model":
                    entropy_model = Entropy_gaussian(channel=num_ch)
                
                entropy_model.load_state_dict(value)
                self.entropy_models[attr_name] = entropy_model

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
    
    @torch.no_grad()
    def save_params_into_ply_file(
        self
    ):
        """Save parameters of Gaussian Splats into .ply file"""
        ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(ply_dir, exist_ok=True)
        ply_file = ply_dir + "/splats.ply"
        save_ply(self.splats, ply_file)
        print(f"Saved parameters of splats into file: {ply_file}.")


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ply_path is not None or cfg.ckpt is not None:
        if cfg.ckpt is not None:
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=True)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            step = ckpts[0]["step"]
        # run eval only

        if cfg.ply_path is not None:
            splats_param = load_ply(cfg.ply_path)
            for k in runner.splats.keys():
                runner.splats[k].data = splats_param[k].data.to(runner.device)
            step = 0
        
        # runner.save_params_into_ply_file()
        runner.eval(step=step)
        # runner.render_traj(step=step)
        if cfg.compression is not None:
            # if cfg.compression == "entropy_coding":
            #     runner.load_entropy_model_from_ckpt(ckpts[0], cfg.entropy_model_type)
            runner.run_compression(step=step)
    # else: # no need for training
    #     runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "png_compression":(
            "Use PngCompression.",
            Config(
                compression="png",
                compression_cfg=CompressionConfig()
            )
        ),
        "x265_compression_rp0":(
            "Use HevcCompression.",
            Config(
                compression="hevc",
                compression_cfg=CompressionConfig(use_sort=True,
                                                  verbose=True,
                                                  qp=4, 
                                                  n_clusters=8192)
            )
        ),
        "x265_compression_rp1":(
            "Use HevcCompression.",
            Config(
                compression="hevc",
                compression_cfg=CompressionConfig(use_sort=True,
                                                  verbose=True,
                                                  qp=10, 
                                                  n_clusters=8192)
            )
        ),
        "x265_compression_rp2":(
            "Use HevcCompression.",
            Config(
                compression="hevc",
                compression_cfg=CompressionConfig(use_sort=True,
                                                  verbose=True,
                                                  qp=16, 
                                                  n_clusters=8192)
            )
        ),
        "x265_compression_rp3":(
            "Use HevcCompression.",
            Config(
                compression="hevc",
                compression_cfg=CompressionConfig(use_sort=True,
                                                  verbose=True,
                                                  qp=22, 
                                                  n_clusters=8192)
            )
        ),
        "x265_compression_debug":(
            "Use HevcCompression.",
            Config(
                compression="hevc",
                compression_cfg=CompressionConfig(
                                                  qp=22, 
                                                  n_clusters=8192)
            )
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

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
