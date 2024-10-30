import json
import math
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import yaml
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
from gsplat.rendering import rasterization
from helper.STG.helper_model import getcolormodel, trbfunction
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from gsplat.strategy.STG_Strategy import STG_Strategy # import densification and pruning strategy that fits STG model
from fused_ssim import fused_ssim
from operator import itemgetter

@dataclass
class Config:
    # Model Params / lp
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: int = 2 #-1
    white_background: bool = False
    veryrify_llff: int = 0
    eval: bool = True
    model: str = "gmodel" # 
    loader: str = "colmap" #
    normalize_world_space: bool = True # Normalize the world space
    
    # Optimization Params / op
    max_steps: int = 30_000
    init_opa: float = 0.1 # Initial opacity of GS
    batch_size: int = 2 # TODO Do not support batch size = 1 for now
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
    duration: int = 50 # 20 # number of frames to train
    ssim_lambda: float = 0.2 # Weight for SSIM loss
    save_steps: List[int] = field(default_factory=lambda: [7_000, 25_000, 30_000]) # Steps to save the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 25_000, 30_000]) # Steps to evaluate the model # 7_000, 30_000
    # eval_steps: List[int] = field(default_factory=lambda: [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 25_000, 30_000])
    desicnt: int = 6
    position_lr_init = 1.6e-4
    scaling_lr = 5e-3
    rotation_lr = 1e-3
    opacity_lr = 5e-2
    trbfs_lr = 0.03
    trbfc_lr = 0.0001
    movelr = 3.5
    omega_lr = 0.0001
    
    tb_every: int = 100 # Dump information to tensorboard every this steps
    # model_path = "/home/czwu/oursSTG_output/model/flame_steak" # dir of output model
    # data_dir: str = "/data/czwu/Neural_3D_Dataset/flame_steak/colmap_0" # modified to fit STG style data loader
    # result_dir: str = "/home/czwu/oursSTG_output/results/flame_steak" # Directory to save results
    model_path = "/home/czwu/oursSTG_output_CPU_3/model/flame_steak" # dir of output model
    data_dir: str = "/data/czwu/Neural_3D_Dataset/flame_steak/colmap_0" # modified to fit STG style data loader
    result_dir: str = "/home/czwu/oursSTG_output_CPU_3/results/flame_steak" # Directory to save results
    ckpt: str = None # Serve as checkpoint, Same as "start_checkpoint" in STG
    lpips_net: str = "alex" # "alex" or "vgg"
    

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
    # quats = torch.rand((N, 4))  # [N, 4]
    quats = torch.zeros((N, 4))
    quats[:, 0] = 1
    # opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    opacities = inverse_sigmoid(0.1 * torch.ones(N,))
    trbf_scale = torch.ones((N, 1))
    times = parser.timestamp 
    times = torch.tensor(times)
    trbf_center = times.contiguous() 
    motion = torch.zeros((N, 9))
    omega = torch.zeros((N, 4))
    
    # rgbdecoder = getcolormodel()
    
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
        params.append(("features_dir", torch.nn.Parameter(colors), 2.5e-3))
        # params.append(("features_dir", torch.nn.Parameter(features_dir), 2.5e-3))
        features_time = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features_time", torch.nn.Parameter(features_time), 2.5e-3)) 
        # colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3)) # feature base # inf not in here?
    
    
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
        parser = Parser(model_path=self.cfg.model_path, source_path=self.cfg.data_dir, duration=cfg.duration, shuffle=False, eval=self.cfg.eval, resolution=cfg.resolution, data_device=cfg.device)
        self.parser = parser
        self.trainset = Dataset(parser=self.parser, split="train")
        self.testset = Dataset(parser=self.parser, split="test")
        
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
        maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# z wrong...
        minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])
        self.maxbounds = [maxx, maxy, maxz]
        self.minbounds = [minx, miny, minz]
        
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
        # TODO Compression Strategy should proceed here, according to GSplat
        
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
        
    def rasterize_splats(
        self,
        timestamp: float, 
        Ks: Tensor,
        width: int,
        height: int,
        basicfunction, 
        rays, 
        camtoworld,
        batch_size,
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
        feature_dir = self.splats["features_dir"] # [N, 3]
        feature_time = self.splats["features_time"] # [N, 3]    
        timestamp = timestamp
        
        trbfdistanceoffset = timestamp * pointtimes - trbfcenter
        trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
        trbfoutput = basicfunction(trbfdistance)
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        opacity = opacities * trbfoutput.squeeze()
        
        tforpoly = trbfdistanceoffset.detach()
        # Calculate Polynomial Motion Trajectory
        means_motion = means + motion[:, 0:3] * tforpoly + motion[:, 3:6] * tforpoly * tforpoly + motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        # Calculate rotations
        rotations = torch.nn.functional.normalize(quats + tforpoly * omega)

        # Calculate feature
        colors_precomp = torch.cat((feature_color, feature_dir, tforpoly * feature_time), dim=1)
        # colors_precomp = feature_color
        
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
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)
        
        max_steps = cfg.max_steps
        init_step = 0
        
        flag = 0
        
        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        
        # organize data accoring to their timestamp, to ensure data in the same batch have the same timestamp
        # TODO This part consumes too much time when duration is high
        # import pdb; pdb.set_trace()
        print("organizing data accoring to their timestamp, this may take a while...")
        cam_num = int(len(self.trainset)/cfg.duration)
        if cfg.batch_size > 1:
            # traincameralist = self.trainset
            traincamdict = {}
            for timeindex in range(cfg.duration): 
                traincamdict[timeindex] = itemgetter(*[timeindex+cfg.duration*i for i in range(cam_num)])(self.trainset)
                # traincamdict[i] = []
                # for j in range(len(self.trainset)):
                #     if self.trainset[j]["timestamp"] == i/cfg.duration:
                #         traincamdict[i].append(self.trainset[j])
        else: 
            # Do not support batch size = 1 for now
            raise ValueError(f"Batch size = 1 is not supported: {cfg.batch_size}")
        print("organizing data complete!")
        
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
        
        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            # Viewer
            # try:
            #     data = next(trainloader_iter)
            # except StopIteration:
            #     trainloader_iter = iter(trainloader)
            #     data = next(trainloader_iter)
            
            timeindex = randint(0, cfg.duration-1)
            # data_set = random.sample(traincamdict[timeindex], cfg.batch_size)
            # cam_num = int(len(self.trainset)/cfg.duration)
            # data_set = random.sample(itemgetter(*[timeindex+cfg.duration*i for i in range(cam_num)])(self.trainset), cfg.batch_size)
            data_set = random.sample(traincamdict[timeindex], cfg.batch_size)
            
            # timeindex = 0
            # data_set = traincamdict[0][0:2]
            
            # TODO Test if the following part is the reason why oursSTG has longer training time
            for i in range(cfg.batch_size):
                if i == 0:
                    Ks = data_set[i]["K"].float().to(device)
                    pixels = data_set[i]["image"].float().to(device)
                    height, width = pixels.shape[0:2]
                    timestamp = data_set[i]["timestamp"]
                    rays = data_set[i]["ray"].float().to(device)
                    camtoworld = data_set[i]["camtoworld"].float().to(device)
                else:
                    Ks = torch.stack((Ks, data_set[i]["K"].float().to(device)), dim=0)
                    pixels = torch.stack((pixels, data_set[i]["image"].float().to(device)), dim=0)
                    rays = torch.stack((rays, data_set[i]["ray"].float().to(device)), dim=0)
                    camtoworld = torch.stack((camtoworld, data_set[i]["camtoworld"].float().to(device)), dim=0)
            rays = rays.squeeze()

            # forward
            renders, alphas, info = self.rasterize_splats(
                # R=R,
                # T=T,
                timestamp=timestamp, # [C]
                Ks=Ks, # [C, 3, 3]
                width=width,
                height=height,
                basicfunction=trbfunction,
                rays=rays, # [C, 6, H, W]
                camtoworld=camtoworld, # [C, 4, 4]
                batch_size=cfg.batch_size,
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
            
            loss.backward()
            desc = f"loss={loss.item():.3f}| "
            pbar.set_description(desc)
            
            # write images (gt and render); output L1 difference map; Only plot the first image in one batch
            if world_rank == 0 and (step == 7_000 or step == 25_000) :
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
          
            # Write TensorBoard here,according to gsplat
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                # self.writer.add_histogram("train/means_GS", self.splats["means"], step)
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.flush()
            
            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
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
                data = {"step": step, "splats": self.splats.state_dict()}
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
            else:
                assert False, "Invalid strategy!" 
                
            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            self.decoder_optimizer.step()
            self.decoder_optimizer.zero_grad(set_to_none=True)
                
            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                # self.render_traj(step)

            # run compression
            # TODO
            
            # Viewer Skipped
            # TODO
               
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        # world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.testset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            R = data['R'].float().to(device)
            T = data['T'].float().to(device) 
            Ks = data["K"].float().to(device)
            pixels = data["image"].float().to(device) # / 255.0
            height, width = pixels.shape[1:3]
            timestamp = data["timestamp"].float().to(device)
            rays = data["ray"].squeeze(0).float().to(device) 
            camtoworld = data['camtoworld'].float().to(device)
            
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
                batch_size=1,
            )
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
                difference = abs(colors - pixels).squeeze().detach().cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/difference_map/train_step{step}_{i:04d}.png",
                    (difference * 255).astype(np.uint8),
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
    def render_traj(self, step: int):
        pass

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
        step = ckpts[0]["step"]
        runner.eval(step=step)

    else:
        runner.train()

if __name__ == "__main__":
    cfg = Config()
    main(cfg)
    
    