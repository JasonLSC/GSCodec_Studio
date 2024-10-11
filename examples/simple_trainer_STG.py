import json
import math
import os
import torch
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from datasets.INVR import Dataset, Parser # This only supports preprocessed Bartender & CBA dataset
from gsplat.rendering import rasterization
from helper.STG.helper_model import getcolormodel, trbfunction
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from gsplat.strategy import STG_Strategy # import densification and pruning strategy that fit STG model

@dataclass
class Config:
    # TODO
    # Model Params / lp
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: int = -1
    white_background: bool = False
    data_device: str = "cuda"
    veryrify_llff: int = 0
    eval: bool = False
    model: str = "gmodel" # 
    loader: str = "colmap" #
    normalize_world_space: bool = True # Normalize the world space
    
    # Optimization Params / op
    iterations: int = 30_000
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
    absgrad: bool = False
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    
    # Pipeline Params / pp
    
    # Others / Parser Params
    data_dir: str = "/data/czwu/Bartender"
    ckpt: str = None # Serve as checkpoint, Same as "start_checkpoint" in STG

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
        # points = torch.from_numpy(parser.points[0]).float()
        # rgbs = torch.from_numpy(parser.points[1]).float()
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
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    trbf_scale = torch.ones((N, 1))
    times = parser.timestamp # May not be correct
    times = torch.tensor(times)
    trbf_center = times.contiguous() # Shape and type may not be correct
    motion = torch.zeros((N, 9))
    omega = torch.zeros((N, 4))
    
    rgbdecoder = getcolormodel()
    
    # TODO 这里的lr最好做成cfg参数传入的形式，而不是直接把数值放在这里
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2), # time-independent spatial opacity
        # The following params' lr are not tested
        ("trbf_scale", torch.nn.Parameter(trbf_scale), 0.03),
        ("trbf_center", torch.nn.Parameter(trbf_center), 0.0001),
        ("motion", torch.nn.Parameter(motion), 1.6e-4 * scene_scale * 0.5 * 3.5),
        ("omega", torch.nn.Parameter(omega), 0.0001),
        ("decoder_params", list(rgbdecoder.parameters()), 0.0001),
    ]
    
    # N * 3 base color (feature base) & N * 3 time-independent feature dir & N * 3 time-dependent feature
    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        # The following params' lr are not tested
        features_dir = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features_dir", torch.nn.Parameter(features_dir), 2.5e-3))
        features_time = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features_time", torch.nn.Parameter(features_time), 2.5e-3)) 
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3)) # feature base
    
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
    return splats, optimizers, rgbdecoder

class Runner:
    """Engine for training and testing."""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Write cfg file: Skipped
        
        # Load data: Training data should contain initial points and colors.
        self.parser_train = Parser(
            data_dir=cfg.data_dir,
            set = "train",
            normalize = cfg.normalize_world_space,
        )
        self.parser_val = Parser(
            data_dir=cfg.data_dir,
            set = "test",
        )
        self.trainset = Dataset(self.parser_train,)
        self.valset = Dataset(self.parser_val,)
        
        # scene scale
        self.scene_scale = self.parser_train.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)
        
        # Initialize Model
        self.splats, self.optimizers, self.rgbdecoder = create_splats_with_optimizers(
            self.parser_train,
            init_opacity=self.cfg.init_opa,
            batch_size=self.cfg.batch_size,
            feature_dim=self.cfg.feature_dim,
            device=self.cfg.device,
            # Model Params 
            # Optimization Params
            # Additional Params?
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        
        # TODO 这个跟create_splats_with_optimizers里面实例化的decoder重复了，需要修改 Experimental
        self.decoder = getcolormodel()
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.0001)
        
        currentxyz = self.splats["means"]
        maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# z wrong...
        minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
        maxbounds = [maxx, maxy, maxz]
        minbounds = [minx, miny, minz]
        
        # Densification Strategy
        # Only support one type of Densification Strategy for now
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
            # revised_opacity=cfg.revised_opacity,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)
        
        # Compression Strategy
        # TODO 压缩最好在完成整个pipeline之后做
        
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
        # TODO viewer最好在完成整个pipeline之后做
        
        
        import pdb; pdb.set_trace()
         
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        timestamp: float, # TODO this param needs to be obtained from the dataset and given into this function
        Ks: Tensor,
        width: int,
        height: int,
        basicfunction, # TODO this param needs to be given into this function, expect input "trbfunction"
        # mode: str = "train", # train or test, specify usage 
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # preprocess splats data
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        trbfcenter = self.splats["trbf_center"] # [N, 1]
        trbfscale = self.splats["trbf_scale"] # [N, 1]
        pointtimes = torch.ones((means.shape[0],1), dtype=means.dtype, requires_grad=False, device="cuda") + 0 # 
        motion = self.splats["motion"] # [N, 9]
        omega = self.splats["omega"] # [N, 4]
        feature_color = self.splats["colors"] # [N, 3]
        feature_dir = self.splats["feature_dir"] # [N, 3]
        feature_time = self.splats["feature_time"] # [N, 3]
        # if mode == "train":
        #     parser = self.parser_train
        # elif mode == "test":
        #     parser = self.parser_val
        # else:
        #     assert False, "Invalid mode input."
        
        # TODO calculate opacity, Eq（7）
        trbfdistanceoffset = timestamp * pointtimes - trbfcenter
        trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
        trbfoutput = basicfunction(trbfdistance)
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        opacity = opacities * trbfoutput
        
        tforpoly = trbfdistanceoffset.detach()
        # Calculate Polynomial Motion Trajectory
        means_motion = means + motion[:, 0:3] * tforpoly + motion[:, 3:6] * tforpoly * tforpoly + motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        # Calculate rotations
        rotations = torch.nn.functional.normalize(quats + tforpoly * omega)
        # Calculate feature
        colors_precomp = torch.cat((feature_color, feature_dir, tforpoly * feature_time), dim=1)

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means_motion, # input position
            quats=rotations, # input Polynomial Rotation
            scales=scales,
            opacities=opacity, # input temporal opacity
            colors=colors_precomp, # input concatenated feature
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            # distributed=self.world_size > 1, # TODO
            **kwargs,
        )
        # Decode
        # TODO viewpoint_camera.rays怎么获得？
        render_colors = self.decoder(render_colors.unsqueeze(0), viewpoint_camera.rays, timestamp) # 1 , 3
        render_colors = render_colors.squeeze(0)
        
        return render_colors, render_alphas, info
    
    def train(self):
        pass
    
    @torch.no_grad()
    def eval(self, step: int):
        pass
    
    

def main(cfg: Config):
    runner = Runner(cfg)
    
    if cfg.ckpt is not None:
        # TODO
        # run eval only
        pass
    else:
        runner.train()

if __name__ == "__main__":
    cfg = Config
    main(cfg)