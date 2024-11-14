import json
import math
import os
import time
from dataclasses import dataclass, field # more info about dataclasses: https://docs.python.org/3/library/dataclasses.html
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import imageio
import nerfview
import numpy as np
import torch
import tqdm
import tyro
import viser
import yaml
# from datasets.colmap import Dataset, Parser
from datasets.INVR_N3D import Parser, Dataset # This only supports preprocessed N3D Dataset
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
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression 
from gsplat.distributed import cli
from gsplat.rendering import rasterization
# from gsplat.strategy import DefaultStrategy, MCMCStrategy # Densification and Pruning Strategy
from gsplat.strategy.SOM_Strategy import SOM_Strategy # Densification and Pruning Strategy
from gsplat.strategy import MCMCStrategy

import torch.nn as nn
from operator import itemgetter
import random
from random import randint
from helper.SOM.helper_model import cont_6d_to_rmat
import roma
from scipy.spatial import KDTree

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None # pngCompression is by far the only compression method available
    # Render trajectory path
    render_traj_path: str = "interp" 

    # Path to the Mip-NeRF 360 dataset
    # data_dir: str = "/data/czwu/gsplat_example/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    # result_dir: str = "/home/czwu/gsplat_output/results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 8 # 8 # 2 
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 15_000 # 30_000 # 15_000 # 7_000 # 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [i for i in range(1_000, 15_001, 1_000)])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [i for i in range(1_000, 15_001, 1_000)])

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
    # Union type; Union[X, Y] means either X or Y.
    strategy: Union[SOM_Strategy, MCMCStrategy] = field(
        default_factory=SOM_Strategy
    )
    # strategy: Union[DefaultStrategy, MCMCStrategy] = field(
    #     default_factory=DefaultStrategy
    # )
    
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
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

    lpips_net: Literal["vgg", "alex"] = "vgg"
    
    # Newly added for Basis & Coefs
    K_coefs: int = 16 # 10 # equals B in Eq(4)
    # model_path = "/home/czwu/oursSOM_output_test/model/flame_steak" # dir of output model
    model_path = "/home/czwu/results_basisAndCoefs/model/cook_spinach_vgg"

    
    # data_dir: str = "/data/czwu/Neural_3D_Dataset/flame_steak/colmap_0" 
    data_dir: str = "/data/czwu/neural_3d/cook_spinach/colmap_0"
    
    # result_dir: str = "/home/czwu/oursSOM_output_test/results/flame_steak" # Directory to save results
    result_dir: str = "/home/czwu/results_basisAndCoefs/results/cook_spinach_vgg"
    
    duration: int = 50 # number of frames to train
    eval: bool = True
    resolution: int = 2 #-1
    device: str = "cuda"
    lr_motion_coefs: float = 1e-2
    lr_rots: float = 1.6e-4
    lr_transls: float = 1.6e-4 
    w_l1_coefs: float = 0.0  
    w_l_sparcity: float = 0.02 
    w_l_rigid: float = 2 
    num_knn: int = 20
    
    
    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        # if isinstance(strategy, DefaultStrategy):
        if isinstance(strategy, SOM_Strategy):
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
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    K_coefs: int = 10,
    lr_motion_coefs: float = 1e-2,
    num_knn: int = 20,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb).float()
        # rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
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
    
    # Initialize per-gaussian coefs
    # coefs = torch.ones((N, K_coefs))
    coefs = torch.rand((N, K_coefs))
    # coefs = coefs / coefs.sum(dim=1, keepdim=True)
    coefs = F.softmax(coefs, dim=-1)
    
    # Calculate KNN index
    print("Calulating KNN...")
    points_np = points.numpy()
    k = num_knn
    tree = KDTree(points_np)
    distances, indices = tree.query(points_np, k=k)
    
    knn_indices = torch.tensor(indices) # [G, k]
    neighbor_weight = np.exp(-2000 * distances)
    neighbor_weight = torch.tensor(neighbor_weight)
    distances = torch.tensor(distances)
    print("Calulation complete.")
    
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("coefs", torch.nn.Parameter(coefs), lr_motion_coefs),
        # ("neighbor_index", torch.nn.Parameter(knn_indices.float()), 0.0), # TODO 这个虽然是为了方便，但是还是不太合理
        # ("neighbor_weight", torch.nn.Parameter(neighbor_weight.float()), 0.0),
        # ("distances", torch.nn.Parameter(distances), 0.0),
    ]
    
    helper_params = {
        "neighbor_index": knn_indices.to(device),
        "neighbor_weight": neighbor_weight.to(device),
        "distances": distances.to(device),
        }
    # helper_params = helper_params.to(device)

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
    
    
    # splats = torch.nn.ParameterDict({n: v for n, v, _ in params if n is not "neighbor_index"}).to(device)
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
        # for name, _, lr in params if not name == "neighbor_index"
    }
    return splats, optimizers, helper_params


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
        
        os.makedirs(cfg.model_path, exist_ok=True)
        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        
        self.render_dir_difference = f"{cfg.result_dir}/renders/difference_map"
        os.makedirs(self.render_dir_difference, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        # self.parser = Parser(
        #     data_dir=cfg.data_dir,
        #     factor=cfg.data_factor,
        #     normalize=cfg.normalize_world_space,
        #     test_every=cfg.test_every,
        # )
        # self.trainset = Dataset(
        #     self.parser,
        #     split="train",
        #     patch_size=cfg.patch_size,
        #     load_depths=cfg.depth_loss,
        # )
        # data = self.trainset.__getitem__(item = 0)
        # self.valset = Dataset(self.parser, split="val")
        parser = Parser(model_path=self.cfg.model_path, source_path=self.cfg.data_dir, duration=cfg.duration, shuffle=False, eval=self.cfg.eval, resolution=cfg.resolution, data_device=cfg.device)
        self.parser = parser
        self.trainset = Dataset(parser=self.parser, split="train")
        self.testset = Dataset(parser=self.parser, split="test")
        
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers, self.helper_params = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            K_coefs=cfg.K_coefs,
            lr_motion_coefs=cfg.lr_motion_coefs,
            num_knn= cfg.num_knn,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        
        # Intialize Basis
        rots = torch.zeros((cfg.K_coefs, cfg.duration, 6))
        rots[:,:,0] = 1; rots[:,:,4] = 1
        transls = torch.zeros((cfg.K_coefs, cfg.duration, 3)) 
        
        # rots = torch.rand((cfg.K_coefs, cfg.duration, 6))
        # transls = torch.rand((cfg.K_coefs, cfg.duration, 3)) / 10
        # transls = torch.zeros((cfg.K_coefs, cfg.duration, 3)) 
        
        # self.basis_params = nn.ParameterDict(
        #     {
        #         "rots": nn.Parameter(rots),
        #         "transls": nn.Parameter(transls),
        #     }
        # )
        basis_params = [
            # name, value, lr
            ("rots", torch.nn.Parameter(rots), cfg.lr_rots),
            ("transls", torch.nn.Parameter(transls), cfg.lr_transls),
        ]

        basis = torch.nn.ParameterDict({n: v for n, v, _ in basis_params}).to(cfg.device)
        self.basis = basis

        BS = cfg.batch_size
        self.basis_optimizers = {
            name: (torch.optim.SparseAdam if cfg.sparse_grad else torch.optim.Adam)(
                [{"params": basis[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, _, lr in basis_params
        }
        
        
        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        
        # if isinstance(self.cfg.strategy, DefaultStrategy):
        if isinstance(self.cfg.strategy, SOM_Strategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

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

    def rasterize_splats(
        self,
        means,
        quats,
        # timestamp: float, 
        Ks: Tensor,
        width: int,
        height: int,
        camtoworld,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # means = self.splats["means"]  # [N, 3]
        means = means  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        # quats = self.splats["quats"]  # [N, 4]
        quats = quats  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree)
        # image_ids = kwargs.pop("image_ids", None)
        # if self.cfg.app_opt:
        #     colors = self.app_module(
        #         features=self.splats["features"],
        #         embed_ids=image_ids,
        #         dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
        #         sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
        #     )
        #     colors = colors + self.splats["colors"]
        #     colors = torch.sigmoid(colors)
        # else:
        #     colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                # if isinstance(self.cfg.strategy, DefaultStrategy)
                if isinstance(self.cfg.strategy, SOM_Strategy)
                else False
            ),
            sh_degree=sh_degree,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            **kwargs,
        )
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )
        # DataLoader
        # trainloader = torch.utils.data.DataLoader(
        #     self.trainset,
        #     batch_size=cfg.batch_size,
        #     shuffle=True,
        #     num_workers=4,
        #     persistent_workers=True,
        #     pin_memory=True,
        # )
        # trainloader_iter = iter(trainloader)
        
        # organize data accoring to their timestamp, to ensure data in the same batch have the same timestamp
        print("organizing data accoring to their timestamp, this may take a while...")
        cam_num = int(len(self.trainset)/cfg.duration)
        if cfg.batch_size > 1:
            # traincameralist = self.trainset
            traincamdict = {}
            for timeindex in range(cfg.duration): 
                traincamdict[timeindex] = itemgetter(*[timeindex+cfg.duration*i for i in range(cam_num)])(self.trainset)
        else: 
            # Do not support batch size = 1 for now
            raise ValueError(f"Batch size = 1 is not supported: {cfg.batch_size}")
        print("organizing data complete!")
        
        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            
            
            # if not cfg.disable_viewer:
            #     while self.viewer.state.status == "paused":
            #         time.sleep(0.01)
            #     self.viewer.lock.acquire()
            #     tic = time.time()

            # try:
            #     data = next(trainloader_iter)
            # except StopIteration:
            #     trainloader_iter = iter(trainloader)
            #     data = next(trainloader_iter)
            # camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            # Ks = data["K"].to(device)  # [1, 3, 3]
            # pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            # num_train_rays_per_step = (
            #     pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            # )
            # image_ids = data["image_id"].to(device)
            # if cfg.depth_loss:
            #     points = data["points"].to(device)  # [1, M, 2]
            #     depths_gt = data["depths"].to(device)  # [1, M]

            # height, width = pixels.shape[1:3]

            # if cfg.pose_noise:
            #     camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            # if cfg.pose_opt:
            #     camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # # sh schedule
            # sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            
            # 改成了batch内部也随机采样时间，而不是采样一个统一的时间
            # Compute transform according to Eq(4)
            timestamp_list = random.sample(range(0, cfg.duration), cfg.batch_size)
            transls = self.basis["transls"][:, timestamp_list]  # (K, B, 3)
            rots = self.basis["rots"][:, timestamp_list]  # (K, B, 6)
            transls = torch.einsum("pk,kni->pni", F.softmax(self.splats["coefs"], dim=-1), transls) # (G, B, 3)
            rots = torch.einsum("pk,kni->pni", F.softmax(self.splats["coefs"], dim=-1), rots)  # (G, B, 6)
            
            rotmats = cont_6d_to_rmat(rots)  # (G, B, 3, 3)
            transfms = torch.cat([rotmats, transls[..., None]], dim=-1)
            # Compute rigid transformation
            means = self.splats["means"]
            
            # dynmf's way
            # means = means.unsqueeze(1).repeat(1,cfg.batch_size,1) + transls
            # SOM's way
            means = torch.einsum(
                "pnij,pj->pni",
                transfms,
                F.pad(means, (0, 1), value=1.0),
            ) # [G, B, 3]
            
            quats = self.splats["quats"]
            quats = roma.quat_xyzw_to_wxyz(
                (
                    roma.quat_product(
                        roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                        roma.quat_wxyz_to_xyzw(quats[:, None]),
                    )
                )
            )
            
            # calculate rigid loss
            # distances_canonical = self.splats["distances"].unsqueeze(2).repeat(1,1,cfg.batch_size)
            distances_canonical = self.helper_params["distances"].unsqueeze(2).repeat(1,1,cfg.batch_size)
            # translate_canonical = (self.splats["means"][self.splats["neighbor_index"].int(),]-self.splats["means"].unsqueeze(1).repeat(1,cfg.num_knn,1)).unsqueeze(2).repeat(1,1,cfg.batch_size,1) # [G, k, B, 3] k = 20
            # translate_t = (means[self.splats["neighbor_index"].int(),]-means.unsqueeze(1).repeat(1,cfg.num_knn,1,1)) # [G, k, B, 3]
            translate_t = (means[self.helper_params["neighbor_index"].int(),]-means.unsqueeze(1).repeat(1,cfg.num_knn,1,1)) # [G, k, B, 3]
            distances_t = torch.sqrt(torch.sum((translate_t) ** 2, dim=3))
            gap_distance = torch.sqrt((distances_t - distances_canonical) ** 2) 
            # weights = self.splats["neighbor_weight"].unsqueeze(2).repeat(1,1,cfg.batch_size)
            weights = self.helper_params["neighbor_weight"].unsqueeze(2).repeat(1,1,cfg.batch_size)
            gap_distance = gap_distance * weights # [G, k, B]
            l_rigid = torch.mean(gap_distance)
              
            data_set = []
            for i in range(cfg.batch_size):
                # timeindex = randint(0, cfg.duration-1)
                timeindex = timestamp_list[i]
                data_set = random.sample(traincamdict[timeindex], 1)[0]
                Ks = data_set["K"].float().to(device)
                pixel = data_set["image"].float().to(device)
                height, width = pixel.shape[0:2]
                # timestamp = data_set["timestamp"]
                # if not timeindex==timestamp:
                #     print("timeindex not equal to timestamp!")
                #     print("timeindex=", timeindex)
                #     print("timestamp=", timestamp)
                #     assert False
                camtoworld = data_set["camtoworld"].float().to(device)
                # else:
                #     Ks = torch.stack((Ks, data_set[i]["K"].float().to(device)), dim=0)
                #     pixels = torch.stack((pixels, data_set[i]["image"].float().to(device)), dim=0)
                #     camtoworld = torch.stack((camtoworld, data_set[i]["camtoworld"].float().to(device)), dim=0)
                
                # forward
                renders, alphas, info = self.rasterize_splats(
                    means=means[:,i,:],
                    quats=quats[:,i,:],
                    Ks=Ks.unsqueeze(0), # [C, 3, 3]
                    width=width,
                    height=height,
                    camtoworld=camtoworld.unsqueeze(0), # [C, 4, 4]
                    # batch_size=cfg.batch_size,
                )
                if renders.shape[-1] == 4:
                    color, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    color, depths = renders, None
                if i == 0:
                    colors = color
                    pixels = pixel.unsqueeze(0)
                else:
                    colors = torch.cat((colors,color),dim=0)
                    # colors = torch.stack((colors, color), dim=1).squeeze()
                    # pixels = torch.stack((pixels, pixel))
                    pixels = torch.cat((pixels,pixel.unsqueeze(0)),dim=0)
            
            
            # if cfg.use_bilateral_grid:
            #     grid_y, grid_x = torch.meshgrid(
            #         (torch.arange(height, device=self.device) + 0.5) / height,
            #         (torch.arange(width, device=self.device) + 0.5) / width,
            #         indexing="ij",
            #     )
            #     grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            #     colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            # if cfg.random_bkgd:
            #     bkgd = torch.rand(1, 3, device=device)
            #     colors = colors + bkgd * (1.0 - alphas)
            
            # Densification and Pruning preprocess
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            
            
            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            
            # Coefs Regularization Loss
            # TODO
            l1_coefs = torch.mean(abs(F.softmax(self.splats["coefs"], dim=-1)))
            
            # Coefs Sparsity Loss
            # TODO
            max_tensor = torch.max(abs(F.softmax(self.splats["coefs"], dim=-1)), dim=1, keepdim=True)[0].expand_as(F.softmax(self.splats["coefs"], dim=-1))
            l_sparcity = torch.mean(abs(F.softmax(self.splats["coefs"], dim=-1))/max_tensor)
            
            # Rigidity Loss
            # TODO
            
            desc = f"loss={loss.item():.3f}| " f"l1_coefs={cfg.w_l1_coefs * l1_coefs:.5f}| " f"l_sparcity={cfg.w_l_sparcity * l_sparcity:.5f}| " f"l_rigid={cfg.w_l_rigid * l_rigid:.5f}| "
            
            base_loss = loss

            # loss = loss + cfg.w_l1_coefs * l1_coefs + cfg.w_l_sparcity * l_sparcity + cfg.w_l_rigid * l_rigid
            loss = loss + cfg.w_l1_coefs * l1_coefs + cfg.w_l_sparcity * l_sparcity 
            
            # if cfg.depth_loss:
            #     # query depths from depth map
            #     points = torch.stack(
            #         [
            #             points[:, :, 0] / (width - 1) * 2 - 1,
            #             points[:, :, 1] / (height - 1) * 2 - 1,
            #         ],
            #         dim=-1,
            #     )  # normalize to [-1, 1]
            #     grid = points.unsqueeze(2)  # [1, M, 1, 2]
            #     depths = F.grid_sample(
            #         depths.permute(0, 3, 1, 2), grid, align_corners=True
            #     )  # [1, 1, M, 1]
            #     depths = depths.squeeze(3).squeeze(1)  # [1, M]
            #     # calculate loss in disparity space
            #     disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
            #     disp_gt = 1.0 / depths_gt  # [1, M]
            #     depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
            #     loss += depthloss * cfg.depth_lambda
            # if cfg.use_bilateral_grid:
            #     tvloss = 10 * total_variation_loss(self.bil_grids.grids)
            #     loss += tvloss

            # # regularizations
            # if cfg.opacity_reg > 0.0:
            #     loss = (
            #         loss
            #         + cfg.opacity_reg
            #         * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
            #     )
            # if cfg.scale_reg > 0.0:
            #     loss = (
            #         loss
            #         + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
            #     )
            
            # ensure no grad for index # TODO 得确认一下这几个参数是不是完全冻结的，梯度有没有正常传递
            # self.optimizers["neighbor_index"].zero_grad(set_to_none=True)
            # self.optimizers["neighbor_weight"].zero_grad(set_to_none=True)
            # self.optimizers["distances"].zero_grad(set_to_none=True)
            
            loss.backward()
            
            pbar.set_description(desc)
 
            # desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            # if cfg.depth_loss:
            #     desc += f"depth loss={depthloss.item():.6f}| "
            # if cfg.pose_opt and cfg.pose_noise:
            #     # monitor the pose error if we inject noise
            #     pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
            #     desc += f"pose err={pose_err.item():.6f}| "
            # pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )
            
            # write images (gt and render)
            if world_rank == 0 and step % 1000 == 0:
                canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                imageio.imwrite(
                    f"{self.render_dir}/train_step{step}.png",
                    (canvas * 255).astype(np.uint8),
                )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/baseloss", base_loss.item(), step)
                self.writer.add_scalar("train/l1_coefs", l1_coefs, step)
                self.writer.add_scalar("train/l_sparcity", l_sparcity, step)
                self.writer.add_scalar("train/l_rigid", l_rigid, step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                # if cfg.depth_loss:
                #     self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                # if cfg.use_bilateral_grid:
                #     self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                    "w_l1_coefs": cfg.w_l1_coefs,
                    "w_l_sparcity": cfg.w_l_sparcity,
                    "w_l_rigid": cfg.w_l_rigid,
                    "num_basis": cfg.K_coefs,
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict(), "basis": self.basis.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            # Densification and Pruning proceed here 
            # if isinstance(self.cfg.strategy, DefaultStrategy):
            if isinstance(self.cfg.strategy, SOM_Strategy):
                flag = self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            # optimize basis as well
            for optimizer in self.basis_optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) 
            
            
             # Calculate KNN index # TODO 这个位置对吗？
            if flag:
                print("Calulating KNN...")
                points_np = self.splats["means"].detach().cpu().numpy()
                k = cfg.num_knn
                tree = KDTree(points_np)
                distances, indices = tree.query(points_np, k=k)
                
                knn_indices = torch.tensor(indices).to(device) # [G, k]
                neighbor_weight = np.exp(-2000 * distances)
                neighbor_weight = torch.tensor(neighbor_weight).to(device)
                distances = torch.tensor(distances).to(device)

                self.helper_params["neighbor_index"] = knn_indices
                self.helper_params["neighbor_weight"] = neighbor_weight
                self.helper_params["distances"] = distances
                print("Calulation complete.")
                
            

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:                
                psnr, _, _ = self.eval(step)

                # self.render_traj(step)

                # save the model with the best eval results
                if not hasattr(self, 'best_psnr') and step>1000:
                    self.best_psnr = psnr
                    data = {"step": step, 
                            "splats": self.splats.state_dict(),
                            "basis": self.basis.state_dict()}
                    torch.save(
                        data, f"{self.ckpt_dir}/ckpt_best_rank{self.world_rank}.pt"
                    )
                elif psnr > getattr(self, 'best_psnr', float('inf')):
                    self.best_psnr = psnr
                    data = {"step": step, 
                            "splats": self.splats.state_dict(),
                            "basis": self.basis.state_dict()}
                    torch.save(
                        data, f"{self.ckpt_dir}/ckpt_best_rank{self.world_rank}.pt"
                    )

            # run compression
            # if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
            #     self.run_compression(step=step)

            # if not cfg.disable_viewer:
            #     self.viewer.lock.release()
            #     num_train_steps_per_sec = 1.0 / (time.time() - tic)
            #     num_train_rays_per_sec = (
            #         num_train_rays_per_step * num_train_steps_per_sec
            #     )
            #     # Update the viewer state.
            #     self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            #     # Update the scene.
            #     self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.testset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        
        # Compute transform according to Eq(4)
        # timestamp_list = random.sample(range(0, cfg.duration), cfg.batch_size)
        timestamp_list = [i for i in range(cfg.duration)]
        transls = self.basis["transls"][:, timestamp_list]  # (K, B, 3)
        rots = self.basis["rots"][:, timestamp_list]  # (K, B, 6)
        transls = torch.einsum("pk,kni->pni", F.softmax(self.splats["coefs"], dim=-1), transls) # (G, B, 3)
        rots = torch.einsum("pk,kni->pni", F.softmax(self.splats["coefs"], dim=-1), rots)  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (G, B, 3, 3)
        transfms = torch.cat([rotmats, transls[..., None]], dim=-1)
            
        # Compute rigid transformation
        means = self.splats["means"]
        means = torch.einsum(
            "pnij,pj->pni",
            transfms,
            F.pad(means, (0, 1), value=1.0),
        ) # [G, B, 3]
        quats = self.splats["quats"]
        quats = roma.quat_xyzw_to_wxyz(
            (
                roma.quat_product(
                    roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                    roma.quat_wxyz_to_xyzw(quats[:, None]),
                )
            )
        )       
        
        for i, data in enumerate(valloader):
            Ks = data["K"].float().to(device)
            pixels = data["image"].float().to(device) # / 255.0
            height, width = pixels.shape[1:3]
            timestamp = data["timestamp"].float().to(device)
            camtoworld = data['camtoworld'].float().to(device)
            # if not i==timestamp:
            #     print("i not equal to timestamp!")
            #     print("i=", i)
            #     print("timestamp=", timestamp)
            #     assert False
            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                means=means[:,i,:],
                quats=quats[:,i,:],
                # timestamp=i,
                camtoworld=camtoworld,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
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
        
        return stats['psnr'], stats['ssim'], stats['lpips']

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
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

        canvas_all = []
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
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

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


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")
    
    # Engine for training and testing
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        runner.basis.load_state_dict(ckpts[0]["basis"])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        # runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

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
    # 在一开始设置的Config基础上，对有需要修改的参数进行修改
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                # strategy=DefaultStrategy(verbose=True),
                strategy=SOM_Strategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
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
