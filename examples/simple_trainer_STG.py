import json
import math
import os
from matplotlib.style import use
import torch
import torch.nn.functional as F
import numpy as np
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union, ContextManager
import yaml
import tyro
import tqdm
import imageio
import matplotlib.pyplot as plt
import random
from random import randint
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.utils.tensorboard import SummaryWriter
# from datasets.INVR import Dataset, Parser # This only supports preprocessed Bartender & CBA dataset
from datasets.INVR_N3D import Parser, Dataset # This only supports preprocessed N3D Dataset

from gsplat import strategy
from helper.STG.helper_model import getcolormodel, trbfunction
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from fused_ssim import fused_ssim

from gsplat.compression import PngCompression, STGPngCompression
from gsplat.rendering import rasterization
from gsplat.strategy import STG_Strategy, Modified_STG_Strategy # import densification and pruning strategy that fits STG model
from gsplat.compression_simulation import STGCompressionSimulation

class ProfilerConfig:
    def __init__(self):
        self.enabled = False
        self.activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        # 基础配置
        self.wait = 4900      # 开始记录前等待的步数
        self.warmup = 50    # 预热步数
        self.active = 30_000    # 实际分析的步数
        # self.repeat = 2    # 重复次数
        # self.skip_first = 10  # 跳过前N步（可选）
        
        # 创建schedule
        self.schedule = self._create_schedule()
        
        # 其他profiler设置
        self.on_trace_ready = torch.profiler.tensorboard_trace_handler('./log/profiler')
        self.record_shapes = True
        self.profile_memory = True
        self.with_stack = True
    
    def _create_schedule(self):

        return torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            # repeat=self.repeat,
            # skip_first=self.skip_first
        )
    
    def update_schedule(self, **kwargs):
        """动态更新schedule参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.schedule = self._create_schedule()


@dataclass
class Config:
    # Model Params / lp
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    downscale_factor: int = 2 #-1
    white_background: bool = False
    veryrify_llff: int = 0
    eval: bool = True
    model: str = "gmodel" # 
    loader: str = "colmap" #
    normalize_world_space: bool = True # Normalize the world space
    
    # Optimization Params / op
    max_steps: int = 30_000
    init_opa: float = 0.1 # Initial opacity of GS
    batch_size: int = 2 
    feature_dim: int = 3
    device: str = "cuda"
    global_scale: float = 1.0 # A global scaler that applies to the scene size related parameters
    prune_opa: float = 0.005 # prune opacity threshold
    grow_grad2d: float = 0.0002 # grad threshold
    grow_scale3d: float = 0.01
    refine_start_iter: int = 500
    refine_stop_iter: int = 9_000 # STG changed this param from 15_000 to 9_000, comprared with 3dgs
    reset_every: int = 3_000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    duration: int = 50 # 20 # number of frames to train
    ssim_lambda: float = 0.2 # Weight for SSIM loss
    save_steps: List[int] = field(default_factory=lambda: [i for i in range(9_000, 75_001, 3_000)]) # Steps to save the model
    eval_steps: List[int] = field(default_factory=lambda: [i for i in range(0, 75_001, 3_000)]) # Steps to evaluate the model # 7_000, 30_000
    # eval_steps: List[int] = field(default_factory=lambda: [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 25_000, 30_000])
    # Number of densification
    desicnt: int = 6 # default: 6
    position_lr_init = 1.6e-4
    scaling_lr = 5e-3
    rotation_lr = 1e-3
    opacity_lr = 5e-2
    trbfs_lr = 0.03
    trbfc_lr = 0.0001
    movelr = 3.5
    omega_lr = 0.0001
    
    tb_every: int = 100 # Dump information to tensorboard every this steps
    model_path: str = "" # dir of output model
    data_dir: str = "" # modified to fit STG style data loader
    result_dir: str = "" # Directory to save results
    ckpt: Optional[List[str]] = None # Serve as checkpoint, Same as "start_checkpoint" in STG
    lpips_net: str = "alex" # "alex" or "vgg"

    # densification strategy
    # strategy: Union[STG_Strategy] = field(
    #     default_factory=DefaultStrategy
    # )
    strategy: Literal["STG_Strategy", "Modified_STG_Strategy"] = "STG_Strategy"

    # Temporal visibility masking
    temp_vis_mask: bool = False

    # Test view
    test_view_id: List[int] = field(default_factory=lambda: [0]) # Neu3DVideo do not need to specify, but INVR needs

    # compression 
    # Name of compression strategy to use
    compression: Optional[Literal["png", "stg"]] = None

    # Enable profiler
    profiler_enabled: bool = False

    # Enable compression simulation
    compression_sim: bool = False
    # Name of quantization simulation strategy to use
    quantization_sim_type: Optional[Literal["round", "noise", "vq"]] = None
    # Enable entropy model
    entropy_model_opt: bool = False
    # Bit-rate distortion trade-off parameter
    rd_lambda: float = 1e-2 # default: 1e-2
    # Steps to enable entropy model into training pipeline
    entropy_steps: Dict[str, int] = field(default_factory=lambda: {"means": -1,
                                                                    "scales": 7000,
                                                                    "quats": 7000,
                                                                    "opacities": -1,
                                                                    "trbf_center": -1,
                                                                    "trbf_scale": -1,
                                                                    "motion": -1, # [N, 9]
                                                                    "omega": -1, # [N, 4]
                                                                    "colors": 7000,
                                                                    "features_dir": 7000,
                                                                    "features_time": 7000,})    

    
    # Enable shN adaptive mask
    shN_ada_mask_opt: bool = False
    # Steps to enable shN adaptive mask
    ada_mask_steps: int = 10_000

    # Enable torch.autograd.detect_anomaly ?
    enable_autograd_detect_anomaly: bool = False
    

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
    feature_dim: Optional[int] = 3,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    # Only considered init_type of sfm for now, which means init_type of random may not work properly
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
        # pass
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    
    # Didn't introduce world_rank and world_size 
    N = points.shape[0]
    # quats = torch.rand((N, 4))  
    quats = torch.zeros((N, 4)) # [N, 4]
    quats[:, 0] = 1
    # opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    opacities = inverse_sigmoid(0.1 * torch.ones(N,))
    trbf_scale = torch.log(torch.ones((N, 1))) # [N, 1]
    times = parser.timestamp 
    times = torch.tensor(times)
    trbf_center = times.contiguous() # [N, 1]
    motion = torch.zeros((N, 9))
    omega = torch.zeros((N, 4))
    
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), cfg.position_lr_init * scene_scale),
        ("scales", torch.nn.Parameter(scales), cfg.scaling_lr),
        ("quats", torch.nn.Parameter(quats), cfg.rotation_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacity_lr), # time-independent spatial opacity
        # The following params' lr are not tested
        ("trbf_scale", torch.nn.Parameter(trbf_scale), cfg.trbfs_lr),
        ("trbf_center", torch.nn.Parameter(trbf_center), cfg.trbfc_lr),
        ("motion", torch.nn.Parameter(motion), cfg.position_lr_init * scene_scale * 0.5 * cfg.movelr),
        ("omega", torch.nn.Parameter(omega), cfg.omega_lr),
        # ("decoder_params", list(rgbdecoder.parameters()), 0.0001),
    ]
    
    # N * 3 base color (feature base) & N * 3 time-independent feature dir & N * 3 time-dependent feature
    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        colors = rgbs
        # features will be used for appearance and view-dependent shading
        # The following params' lr are not tested
        features_dir = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features_dir", torch.nn.Parameter(colors), 2.5e-3)) # TODO: Why use color rather than features_dir?
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3)) # feature base 

        features_time = torch.zeros(N, feature_dim)  # [N, feature_dim]
        params.append(("features_time", torch.nn.Parameter(features_time), 2.5e-3)) 
    
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size
    
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    # return splats, optimizers, rgbdecoder
    return splats, optimizers

class Runner:
    """Engine for training and testing."""
    def __init__(self, cfg: Config) -> None:
        # only enable when debug!!
        if cfg.enable_autograd_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.cfg = cfg
        # Write cfg file: Skipped
        self.device = self.cfg.device
        
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
        parser = Parser(model_path=self.cfg.model_path, source_path=self.cfg.data_dir, duration=cfg.duration, 
                        shuffle=False, eval=self.cfg.eval, downscale_factor=cfg.downscale_factor, data_device='cpu', test_view_id=cfg.test_view_id)
        self.parser = parser
        self.trainset = Dataset(parser=self.parser, split="train", num_views=cfg.batch_size, use_fake_length=True, fake_length=cfg.max_steps+100)
        self.testset = Dataset(parser=self.parser, split="test", num_views=len(cfg.test_view_id))

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=True,
            num_workers=16,
            # persistent_workers=True,
            pin_memory=True,
            # collate_fn=collate_fn
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            # persistent_workers=True,
            pin_memory=True,
            # collate_fn=collate_fn
        )
        # scene scale
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)
        
        # Initialize Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_opacity=self.cfg.init_opa,
            batch_size=self.cfg.batch_size,
            feature_dim=self.cfg.feature_dim,
            device=self.cfg.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        
        self.decoder = getcolormodel().to(cfg.device)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.0001)
        
        currentxyz = self.splats["means"]
        maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])
        minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
        self.maxbounds = [maxx, maxy, maxz]
        self.minbounds = [minx, miny, minz]
        
        # Densification Strategy
        # Only support one type of Densification Strategy for now
        if cfg.strategy == "STG_Strategy":
            self.strategy = STG_Strategy(
                verbose=True, 
                prune_opa=cfg.prune_opa, 
                grow_grad2d=cfg.grow_grad2d, 
                grow_scale3d=cfg.grow_scale3d, 
                # prune_scale3d=cfg.prune_scale3d,
                # refine_scale2d_stop_iter=4000, # splatfacto behavior 
                refine_start_iter=cfg.refine_start_iter, 
                refine_stop_iter=cfg.refine_stop_iter, 
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                pause_refine_after_reset=cfg.pause_refine_after_reset
                # revised_opacity=cfg.revised_opacity,
            )
        elif cfg.strategy == "Modified_STG_Strategy":
            self.strategy = Modified_STG_Strategy(
                verbose=True, 
                prune_opa=cfg.prune_opa, 
                grow_grad2d=cfg.grow_grad2d, 
                grow_scale3d=cfg.grow_scale3d, 
                # prune_scale3d=cfg.prune_scale3d,
                # refine_scale2d_stop_iter=4000, # splatfacto behavior 
                refine_start_iter=cfg.refine_start_iter, 
                refine_stop_iter=cfg.refine_stop_iter, 
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                pause_refine_after_reset=cfg.pause_refine_after_reset,
                temp_vis_mask=cfg.temp_vis_mask
                # revised_opacity=cfg.revised_opacity,
            )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        
        # Compression Strategy
        # TODO Compression Strategy should proceed here, according to GSplat
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            elif cfg.compression == "stg":
                self.compression_method = STGPngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        if cfg.compression_sim:
            # TODO: bad impl. 
            # cap_max = cfg.strategy.cap_max if cfg.strategy.cap_max is not None else None
            self.compression_sim_method = STGCompressionSimulation(cfg.quantization_sim_type,
                                                    cfg.entropy_model_opt, 
                                                    cfg.entropy_steps, 
                                                    self.device, 
                                                    cfg.shN_ada_mask_opt,
                                                    cfg.ada_mask_steps,)
            if cfg.entropy_model_opt:
                selected_key = min((k for k, v in cfg.entropy_steps.items() if v >= 0), key=lambda k: cfg.entropy_steps[k])
                self.entropy_min_step = cfg.entropy_steps[selected_key]
        
        # Profiler
        self.profiler: Optional[torch.profiler.profile] = None
        self.profiler_config = ProfilerConfig()
        if cfg.profiler_enabled:
            self.profiler_config.enabled = True

        # Ignored initializing pose optimization
        # Ignored initializing appearance optimization
        # Ignored bilateral grid initialization
        
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
        # TODO Viewer should proceed here, according to GSplat

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
        timestamp: float, 
        Ks: Tensor,
        width: int,
        height: int,
        basicfunction, 
        rays, 
        camtoworld,
        temp_vis_mask = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        if not self.cfg.compression_sim:
            # preprocess splats data
            means = self.splats["means"]  # [N, 3]
            quats = self.splats["quats"]  # [N, 4]
            scales = torch.exp(self.splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

            trbfcenter = self.splats["trbf_center"] # [N, 1]
            trbfscale = torch.exp(self.splats["trbf_scale"]) # [N, 1]
            
            motion = self.splats["motion"] # [N, 9]
            omega = self.splats["omega"] # [N, 4]
            feature_color = self.splats["colors"] # [N, 3] 
            feature_dir = self.splats["features_dir"] # [N, 3]
            feature_time = self.splats["features_time"] # [N, 3]    
        else:
            # preprocess splats data
            means = self.comp_sim_splats["means"]  # [N, 3]
            quats = self.comp_sim_splats["quats"]  # [N, 4]
            scales = torch.exp(self.comp_sim_splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.comp_sim_splats["opacities"])  # [N,]

            trbfcenter = self.comp_sim_splats["trbf_center"] # [N, 1]
            trbfscale = torch.exp(self.comp_sim_splats["trbf_scale"]) # [N, 1], log domain
            
            motion = self.comp_sim_splats["motion"] # [N, 9]
            omega = self.comp_sim_splats["omega"] # [N, 4]
            feature_color = self.comp_sim_splats["colors"] # [N, 3] 
            feature_dir = self.comp_sim_splats["features_dir"] # [N, 3]
            feature_time = self.comp_sim_splats["features_time"] # [N, 3]    
        
        pointtimes = torch.ones((means.shape[0],1), dtype=means.dtype, requires_grad=False, device="cuda") + 0 # 
        timestamp = timestamp
        
        trbfdistanceoffset = timestamp * pointtimes - trbfcenter
        trbfdistance =  trbfdistanceoffset / (math.sqrt(2) * trbfscale)
        trbfoutput = basicfunction(trbfdistance)           

        # opacity decay 
        opacity = opacities * trbfoutput.squeeze()
        
        # Question: Why detach
        tforpoly = trbfdistanceoffset.detach()
        # Calculate Polynomial Motion Trajectory
        means_motion = means + motion[:, 0:3] * tforpoly + motion[:, 3:6] * tforpoly * tforpoly + motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        # Calculate rotations
        rotations = torch.nn.functional.normalize(quats + tforpoly * omega)

        # Calculate feature
        colors_precomp = torch.cat((feature_color, feature_dir, tforpoly * feature_time), dim=1)

        # Filter out unvisible splats at this timestamp. "mask == 1" means visible, otherwise unvisible.
        if temp_vis_mask:
            t_vis_mask = trbfoutput.squeeze() > 0.05
            means_motion = means_motion[t_vis_mask]
            rotations = rotations[t_vis_mask]
            scales = scales[t_vis_mask]
            opacity = opacity[t_vis_mask]
            colors_precomp = colors_precomp[t_vis_mask]

            num_t_vis_mask = t_vis_mask.sum()
            num_all_splats = t_vis_mask.shape[0]
        
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means_motion, # input position
            quats=rotations, # input Polynomial Rotation
            scales=scales,
            opacities=opacity, # input temporal opacity
            colors=colors_precomp, # input concatenated feature
            viewmats=torch.linalg.inv(camtoworld),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            # distributed=self.world_size > 1, # TODO
            **kwargs,
        )

        # (Opt.) modify per-Gaussian related data in "info", except "info['means2d']"
        if temp_vis_mask:
            for name, v in info.items():
                if isinstance(v, torch.Tensor) and name in ["radii", "depths", "conics", "opacities"]:
                    new_shape = list(v.shape)
                    new_shape[1] = num_all_splats
                    new_tensor = torch.zeros(new_shape, dtype=v.dtype, device=v.device)
                    new_tensor[:, t_vis_mask] = v

                    info[name] = new_tensor
            
            info.update(
                {"t_vis_mask": t_vis_mask}
            )

        # Decode
        render_colors = render_colors.permute(0,3,1,2)
        render_colors = self.decoder(render_colors, rays, timestamp) # 1 , 3
        render_colors = render_colors.permute(0,2,3,1)

        # pixels(GT) shape: [1, H, W, 3]
        return render_colors, render_alphas, info
    
    def train(self):
        cfg = self.cfg
        device = self.device
        
        world_rank = 0 # TODO reserved for future work
        self.world_rank = world_rank
        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)
        
        max_steps = cfg.max_steps
        init_step = 0
        
        flag = 0
        
        ### used in gsplat, but no need for now in STG
        # schedulers = [
        #     # means has a learning rate schedule, that end at 0.01 of the initial value
        #     torch.optim.lr_scheduler.ExponentialLR(
        #         self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        #     ),
        # ]
        
        # organize data accoring to their timestamp, to ensure data in the same batch have the same timestamp
        # TODO This part consumes too much time when duration is high
        # import pdb; pdb.set_trace()
        # print("organizing data accoring to their timestamp, this may take a while...")
        # cam_num = int(len(self.trainset)/cfg.duration)
        
        # if True:
        #     if cfg.batch_size > 1:
        #         # traincameralist = self.trainset
        #         traincamdict = {}
        #         for timeindex in range(cfg.duration): 
        #             traincamdict[timeindex] = itemgetter(*[timeindex+cfg.duration*i for i in range(cam_num)])(self.trainset)
        #             # traincamdict[i] = []
        #             # for j in range(len(self.trainset)):
        #             #     if self.trainset[j]["timestamp"] == i/cfg.duration:
        #             #         traincamdict[i].append(self.trainset[j])
        #     else: 
        #         # Do not support batch size = 1 for now
        #         raise ValueError(f"Batch size = 1 is not supported: {cfg.batch_size}")
        #     print("organizing data complete!")
        
        # DataLoader for batchsize=1
        # trainloader = torch.utils.data.DataLoader(
        #     self.trainset,
        #     batch_size=cfg.batch_size,
        #     shuffle=False, # True
        #     num_workers=4,
        #     persistent_workers=True,
        #     pin_memory=True,
        # )
        # trainloader_iter = iter(trainloader)
        
        with self.get_profiler(self.writer) as prof:
            self.profiler = prof if self.profiler_config.enabled else None

            # Training loop.
            global_tic = time.time()
            pbar = tqdm.tqdm(range(init_step, max_steps))
            step = 0
            for batch in self.trainloader:
                step += 1
                pbar.update(1)
                if step > max_steps:
                    pbar.close()
                    break
                
                # get batch data
                pixels = batch["image"][0].to(device)
                Ks, rays, camtoworld = batch["K"][0].to(device), batch["ray"][0].to(device), batch["camtoworld"][0].to(device)
                timestamp = batch['timestamp'][0].to(device).to(torch.float32)
                num_views, height, width, _ = pixels.shape

                # compression simulation
                if cfg.compression_sim:
                    self.comp_sim_splats, self.esti_bits_dict = self.compression_sim_method.simulate_compression(self.splats, step)
                
                # forward
                renders, alphas, info = self.rasterize_splats(
                    timestamp=timestamp, # [C]
                    Ks=Ks, # [C, 3, 3]
                    width=width,
                    height=height,
                    basicfunction=trbfunction,
                    rays=rays, # [C, 6, H, W]
                    camtoworld=camtoworld, # [C, 4, 4]
                    temp_vis_mask=self.cfg.temp_vis_mask
                )

                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None
                
                # Densification and Pruning preprocess
                self.strategy.step_pre_backward(
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

                # entropy constraint
                if cfg.entropy_model_opt and step>self.entropy_min_step:
                    total_esti_bits = 0
                    for n, n_step in cfg.entropy_steps.items():
                        if step > n_step and self.esti_bits_dict[n] is not None:
                            # maybe give different params with different weights
                            total_esti_bits += torch.sum(self.esti_bits_dict[n]) / self.esti_bits_dict[n].numel()
                        # else:
                        #     total_esti_bits += 0

                    loss = (
                        loss
                        + cfg.rd_lambda * total_esti_bits
                    )
                
                # tmp workaround
                loss_show = loss.detach().cpu()

                loss.backward()

                desc = f"loss={loss_show.item():.3f}| "
                pbar.set_description(desc)
                
                # write images (gt and render); output L1 difference map; Only plot the first image in one batch
                if world_rank == 0 and step in cfg.save_steps:
                    canvas = torch.cat([pixels[0:1,...], colors[0:1,...]], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    imageio.imwrite(
                        f"{self.render_dir}/train_step{step}.png",
                        (canvas * 255).astype(np.uint8),
                    )
                    difference = abs(colors[0:1,...] - pixels[0:1,...]).squeeze().detach().cpu().numpy()
                    imageio.imwrite(
                        f"{self.render_dir}/difference_map/train_step{step}.png",
                        (difference * 255).astype(np.uint8),
                    )
            
                # TensorBoard
                if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    self.writer.add_scalar("train/loss", loss.item(), step)
                    self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                    self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                    self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                    # self.writer.add_histogram("train/means_GS", self.splats["means"], step)
                    self.writer.add_scalar("train/mem", mem, step)
                    # if cfg.entropy_model_opt and step>self.entropy_min_step:
                    #     self.writer.add_histogram("train_hist/quats", self.splats["quats"], step)
                    #     # self.writer.add_histogram("train_hist/scales", self.splats["scales"], step)
                    #     # self.writer.add_histogram("train_hist/opacities", self.splats["opacities"], step)
                    #     self.writer.add_histogram("train_hist/colors", self.splats["colors"], step)
                    #     self.writer.add_histogram("train_hist/features_dir", self.splats["features_dir"], step)
                    #     self.writer.add_histogram("train_hist/features_time", self.splats["features_time"], step)
                    #     if total_esti_bits > 0:
                    #         self.writer.add_scalar("train/bpp_loss", total_esti_bits.item(), step)
                    self.writer.flush()
                
                # save checkpoint before updating the model
                if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                    # train log
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    stats = {
                        "mem": mem,
                        "ellipse_time": time.time() - global_tic,
                        "num_GS": len(self.splats["means"]),
                    }
                    print("Step: ", step, stats)
                    with open(
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                        "w",
                    ) as f:
                        json.dump(stats, f)

                    # save checkpoint
                    data = {"step": step, 
                            "splats": self.splats.state_dict(),
                            "decoder": self.decoder.state_dict()}
                    torch.save(
                        data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                    )
                
                # Densification and Pruning proceed here
                if isinstance(self.strategy, STG_Strategy):
                    flag = self.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                        flag=flag,
                        desicnt=cfg.desicnt,
                        maxbounds=self.maxbounds,
                        minbounds=self.minbounds,
                    )
                elif isinstance(self.strategy, Modified_STG_Strategy):
                    flag = self.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                        flag=flag,
                        desicnt=cfg.desicnt,
                        maxbounds=self.maxbounds,
                        minbounds=self.minbounds,
                    )                    
                else:
                    assert False, "Invalid strategy!" 
                    
                # optimize
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                self.decoder_optimizer.step()
                self.decoder_optimizer.zero_grad(set_to_none=True)
                # (optional) entropy model params. optimize
                if cfg.entropy_model_opt and step > self.entropy_min_step:
                    for name, optimizer in self.compression_sim_method.entropy_model_optimizers.items():
                        if optimizer is not None:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                
                self.step_profiler()

                # torch.cuda.empty_cache()
                    
                # eval the full set
                if step in [i - 1 for i in cfg.eval_steps]:
                    psnr, _, _ = self.eval(step)

                    # self.render_traj(step)

                    # save the model with the best eval results
                    if not hasattr(self, 'best_psnr') and step>10000:
                        self.best_psnr = psnr
                        data = {"step": step, 
                                "splats": self.splats.state_dict(),
                                "decoder": self.decoder.state_dict()}
                        torch.save(
                            data, f"{self.ckpt_dir}/ckpt_best_rank{self.world_rank}.pt"
                        )
                    elif psnr > getattr(self, 'best_psnr', float('inf')):
                        self.best_psnr = psnr
                        data = {"step": step, 
                                "splats": self.splats.state_dict(),
                                "decoder": self.decoder.state_dict()}
                        torch.save(
                            data, f"{self.ckpt_dir}/ckpt_best_rank{self.world_rank}.pt"
                        )

                # run compression
                # TODO
                
                # Viewer Skipped
                # TODO

                # memory management
                self.memory_manage(step)
    
    @torch.no_grad()
    def memory_manage(self, step: int):
        # delete intermeidate variables
        del self.comp_sim_splats
        del self.esti_bits_dict
        if step % 200 == 0:
            torch.cuda.empty_cache()


    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = 0 # TODO: bad hard code, to fix
        # world_rank = self.world_rank
        # world_size = self.world_size

        ellipse_time = 0
        metrics = defaultdict(list)
        pbar = tqdm.tqdm(range(0, len(self.testloader)))

        # save path
        eval_save_path = f"{self.render_dir}/{stage}_step{step}"
        os.makedirs(eval_save_path, exist_ok=True)

        ## init writer(s) based on num. of test views
        writers = [imageio.get_writer(f"{eval_save_path}/{stage}_step{step}_testv{i}.mp4", fps=30, quality=10) for i in range(len(cfg.test_view_id))]
        
        for t_idx, batch in enumerate(self.testloader): # t_idx
            
            Ks = batch["K"][0].float().to(device)
            pixels = batch["image"][0].float().to(device) # / 255.0
            num_views, height, width, _ = pixels.shape
            timestamp = batch["timestamp"][0].float().to(device)
            rays = batch["ray"][0].float().to(device) 
            camtoworld = batch['camtoworld'][0].float().to(device)
            
            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                timestamp=timestamp,
                Ks=Ks,
                width=width,
                height=height,
                basicfunction=trbfunction,
                rays=rays,
                camtoworld=camtoworld,
            )
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0) # colors: [N, H, W, C]
            canvas_list = [pixels, colors]
            
            desc = ""
            if world_rank == 0:
                try:
                    import fpnge
                    canvases = torch.cat(canvas_list, dim=2) # canvas: [N, H, 2*W, C]
                    for i in range(canvases.shape[0]): # loop on test views
                        # save side-by-side comparison
                        canvas = canvases[i].contiguous().cpu().numpy()
                        canvas = (canvas * 255).astype(np.uint8)
                        # new version - fpng
                        
                        png = fpnge.fromNP(canvas) # fpnge needs tensor in order as [H,W,C]
                        # with open(f"{self.render_dir}/{stage}_step{step}_{t_idx:04d}_testv{i}.png", 'wb') as f:
                        #     f.write(png)
                        with open(f"{eval_save_path}/sidebyside_testv{i}_fid{t_idx:04d}.png", 'wb') as f:
                            f.write(png)

                        # save gt
                        gt = pixels[i].contiguous().cpu().numpy()
                        gt = (gt * 255).astype(np.uint8)
                        png = fpnge.fromNP(gt)
                        with open(f"{eval_save_path}/gt_testv{i}_fid{t_idx:04d}.png", 'wb') as f:
                            f.write(png)
                        
                        # save rendered
                        rendered = colors[i].contiguous().cpu().numpy()
                        rendered = (rendered * 255).astype(np.uint8)
                        png = fpnge.fromNP(rendered)
                        with open(f"{eval_save_path}/rendered_testv{i}_fid{t_idx:04d}.png", 'wb') as f:
                            f.write(png)

                except:
                    # original version - imageio
                    canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                    canvas = (canvas * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_{t_idx:04d}.png",
                        canvas,
                    )

                # save rendered test-view videos
                for i in range(colors.shape[0]):
                    color = colors[i].cpu().numpy()
                    color = (color * 255).astype(np.uint8)
                    writers[i].append_data(color)

                # write difference image
                # difference = abs(colors - pixels).squeeze().detach().cpu().numpy()
                # imageio.imwrite(
                #     f"{self.render_dir}/difference_map/{stage}_step{step}_{t_idx:04d}.png",
                #     (difference * 255).astype(np.uint8),
                # )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                desc += f"PSNR={metrics['psnr'][-1]:.2f}| SSIM={metrics['ssim'][-1]:.4f}| LPIPS={metrics['lpips'][-1]:.4f}| "
            pbar.set_description(desc)
            pbar.update(1)
        pbar.close()

        for i, writer in enumerate(writers):
            writer.close()
            print(f"Video saved to {self.render_dir}/{stage}_step{step}_testv{i}.mp4")

        if world_rank == 0:
            ellipse_time /= len(self.testloader)

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
    
    def get_rays(self, R, T, focal_length_x, focal_length_y, width, height):
        '''
        R: c2w
        T: w2c

        '''

        from helper.STG.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY).transpose(0,1)
        camera_center = world_view_transform.inverse()[3, :3]

        
        projectinverse = projection_matrix.T.inverse()
        camera2wold = world_view_transform.T.inverse()
        from kornia import create_meshgrid
        pixgrid = create_meshgrid(height, width, normalized_coordinates=False, device="cpu")[0]
        pixgrid = pixgrid  # H,W,
        
        xindx = pixgrid[:,:,0] # x 
        yindx = pixgrid[:,:,1] # y
    
        from helper.STG.helper_model import pix2ndc
        ndcy, ndcx = pix2ndc(yindx, height), pix2ndc(xindx, width)
        ndcx = ndcx.unsqueeze(-1)
        ndcy = ndcy.unsqueeze(-1)# * (-1.0)
        
        ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

        projected = ndccamera @ projectinverse.T 
        diretioninlocal = projected / projected[:,:,3:] #v 


        direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
        rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

        
        rayo = camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)
        rayd = rays_d.permute(2, 0, 1).unsqueeze(0)   

        rays = torch.cat([rayo, rayd], dim=1)

        return rays

    
    @torch.no_grad()
    def render_traj(self, step: int, stage: str = "val"):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")

        cfg = self.cfg
        device = self.device

        timestamps = torch.from_numpy(np.array([i/len(self.testset.scene_by_t) for i in range(len(self.testset.scene_by_t))])).float().to(device)
        Ks = self.testset[0]["K"].float().to(device)
        pixels = self.testset[0]["image"].float().to(device)
        num_views, height, width, _ = pixels.shape

        camtoworld = self.testset[0]["camtoworld"].float().to(device)

        # from c2w to rays
        # R = camtoworld[0, :3, :3].cpu().numpy()
        # T = torch.inverse(camtoworld)[0, :3, -1].cpu().numpy()
        # rays = self.get_rays(R, T, Ks[0,0,0], Ks[0,1,1], width, height).float().to(device)

        # get v4, v5 for interp
        v4_fr0_global_id = 150
        v5_fr0_global_id = 200
        v4_c2w = torch.from_numpy(self.trainset.camtoworld[v4_fr0_global_id])
        v5_c2w = torch.from_numpy(self.trainset.camtoworld[v5_fr0_global_id])
        # v4_c2w = self.trainset[200]["camtoworld"].float().to(device)
        # v5_c2w = self.trainset[250]["camtoworld"].float().to(device)

        v4_w2c = torch.inverse(v4_c2w)
        v5_w2c = torch.inverse(v5_c2w)

        def get_c2w(time, v4_w2c, v5_w2c):
            R1 = v4_w2c[:3, :3].cpu().numpy()
            T1 = v4_w2c[:3, -1].cpu().numpy()
            R2 = v5_w2c[:3, :3].cpu().numpy()
            T2 = v5_w2c[:3, -1].cpu().numpy()

            from helper.STG.posetrace_utils import interpolate_camera_poses2, qvec2rotmat
            q,t = interpolate_camera_poses2(R1, T1, R2, T2, time % 1)

            R = qvec2rotmat(q) # w2c
            T = np.array(t) # w2c

            # R_T = R.transpose()

            w2c = np.zeros([4,4])
            w2c[:3, :3] = R
            w2c[:3, -1] = T
            w2c[3, 3] = 1

            c2w = np.linalg.inv(w2c)

            return c2w



        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/{stage}_traj_{step}.mp4", fps=30)

        for t_idx, timestamp in tqdm.tqdm(enumerate(timestamps), desc="Rendering trajectory"):

            camtoworld = torch.from_numpy(get_c2w(timestamp.cpu().numpy(), v5_w2c, v4_w2c)).float().to(device)
            camtoworld = camtoworld.unsqueeze(0)

            R = camtoworld[0, :3, :3].cpu().numpy()
            T = torch.inverse(camtoworld)[0, :3, -1].cpu().numpy()
            rays = self.get_rays(R, T, Ks[0,0,0], Ks[0,1,1], width, height).float().to(device)

            colors, _, _ = self.rasterize_splats(
                timestamp=timestamp,
                Ks=Ks,
                width=width,
                height=height,
                basicfunction=trbfunction,
                rays=rays,
                camtoworld=camtoworld,
            )

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            # try:
            #     # new version - fpng
            #     # canvas = torch.cat(canvas_list, dim=2).squeeze(0).contiguous().cpu().numpy() # fpnge needs [H,W,C]
            #     # import pdb; pdb.set_trace()
            #     canvas = canvas_list[-1].squeeze(0).contiguous().cpu().numpy()
            #     canvas = (canvas * 255).astype(np.uint8)
            #     import fpnge
            #     png = fpnge.fromNP(canvas)
            #     with open(f"{self.render_dir}/{stage}_step{step}_{t_idx:04d}.png", 'wb') as f:
            #         f.write(png)

            #     canvas = canvas_list[0].squeeze(0).contiguous().cpu().numpy()
            #     canvas = (canvas * 255).astype(np.uint8)
            #     import fpnge
            #     png = fpnge.fromNP(canvas)
            #     with open(f"{self.render_dir}/{stage}_step{step}_{t_idx:04d}_gt.png", 'wb') as f:
            #         f.write(png)

            # except:
            #     # original version - imageio
            #     canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            #     canvas = (canvas * 255).astype(np.uint8)
            #     imageio.imwrite(
            #         f"{self.render_dir}/{stage}_step{step}_{t_idx:04d}.png",
            #         canvas,
            #     )

            canvas = canvas_list[1].squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/{stage}_traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = 0 # hard code for now

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)
        
        # visualize param. distribution
        self.run_param_distribution_vis(self.splats, save_dir=f"{cfg.result_dir}/visualization/raw")
        
        self.compression_method.compress(compress_dir, self.splats)
        torch.save(self.decoder.state_dict(), os.path.join(compress_dir, 'decoder.pth'))
        # self.run_param_distribution_vis(self.splats, save_dir=f"{cfg.result_dir}/visualization/log_transform")

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        decoder_state_dict = torch.load(os.path.join(compress_dir, 'decoder.pth'))
        
        # visualize param. distribution
        self.run_param_distribution_vis(splats_c, save_dir=f"{cfg.result_dir}/visualization/quant")

        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.decoder.load_state_dict(decoder_state_dict)

        self.eval(step=step, stage="compress")

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

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def main(cfg: Config):
    runner = Runner(cfg)
    
    if cfg.ckpt is not None:
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        runner.decoder.load_state_dict(ckpts[0]["decoder"])
        step = ckpts[0]["step"]
        print(f"Evaluate ckpt saved at step {step}")
        # runner.render_traj(step=step)
        runner.eval(step=step)
        if cfg.compression is not None:
            print(f"Compress ckpt saved at step {step}")
            runner.run_compression(step=step)

    else:
        runner.train()

if __name__ == "__main__":
    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Spacetime Gaussians training from the original paper.",
            Config(
            ),
        ),
        "compression_sim": (
            "Spacetime Gaussians training and compression simulation",
            Config(
                compression_sim = True,
                quantization_sim_type = "round",
                # Placeholders for entropy constraint and ada mask
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)    

    main(cfg)
    
    