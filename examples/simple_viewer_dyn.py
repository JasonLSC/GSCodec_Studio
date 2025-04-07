"""A simple example to render a variant of Dynamic Gaussian Splats

```bash
python examples/simple_viewer_dyn.py --ckpt {path_to_ckpt.pt}
```
"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

def trbfunction(x): 
    return torch.exp(-1*x.pow(2))

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c


class DynGSRenderer:
    def __init__(self, args):
        
        splats = torch.load(args.ckpt[0], map_location="cuda")["splats"]

        self.means = splats["means"]  # [N, 3]
        self.quats = splats["quats"]  # [N, 4]
        self.scales = torch.exp(splats["scales"])  # [N, 3]
        self.opacities = torch.sigmoid(splats["opacities"])  # [N,]

        self.trbfcenter = splats["trbf_center"] # [N, 1]
        self.trbfscale = torch.exp(splats["trbf_scale"]) # [N, 1]
        
        self.motion = splats["motion"] # [N, 9]
        self.omega = splats["omega"] # [N, 4]
        self.feature_color = splats["colors"] # [N, 3] 
        self.feature_dir = splats["features_dir"] # [N, 3]
        self.feature_time = splats["features_time"] # [N, 3]

        self.device = self.means.device

        if args.backend == "gsplat":
            self.rasterization_fn = rasterization
    
    def slice_dyngs_to_3dgs(self, timestamp):
        pointtimes = torch.ones((self.means.shape[0],1), dtype=self.means.dtype, requires_grad=False, device="cuda") + 0 # 
        timestamp = timestamp
        
        trbfdistanceoffset = timestamp * pointtimes - self.trbfcenter
        trbfdistance =  trbfdistanceoffset / (math.sqrt(2) * self.trbfscale)
        trbfoutput = trbfunction(trbfdistance)           

        # opacity decay 
        opacity = self.opacities * trbfoutput.squeeze()
        
        tforpoly = trbfdistanceoffset.detach()
        # Calculate Polynomial Motion Trajectory
        means_motion = self.means + self.motion[:, 0:3] * tforpoly + self.motion[:, 3:6] * tforpoly * tforpoly + self.motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        # Calculate rotations
        rotations = torch.nn.functional.normalize(self.quats + tforpoly * self.omega)

        # Calculate feature
        # colors_precomp = torch.cat((feature_color, feature_dir, tforpoly * feature_time), dim=1)
        colors_precomp = self.feature_color        

        return means_motion, rotations, self.scales, opacity, colors_precomp
    
    def render(self, cameraHandle, timestamp=0, render_mode="RGB"):
        means_t, quats_t, scales_t, opa_t, colors_t = self.slice_dyngs_to_3dgs(timestamp)

        c2w = get_c2w(cameraHandle)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        viewmat = c2w.inverse()

        # W = 1920
        # H = int(W/cameraHandle.aspect)
        # focal_x = W/2/np.tan(cameraHandle.fov/2)
        # focal_y = H/2/np.tan(cameraHandle.fov/2)

        W, H = 1920, 1080
        focal_length = H / 2.0 / np.tan(cameraHandle.fov / 2.0)
        focal_x = focal_length
        focal_y = focal_length
        K = np.array(
                [
                    [focal_x, 0.0, W / 2.0],
                    [0.0, focal_y, H / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alphas, meta = self.rasterization_fn(
            means_t,  # [N, 3]
            quats_t,  # [N, 4]
            scales_t,  # [N, 3]
            opa_t,  # [N]
            colors_t,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            W,
            H,
            # sh_degree=sh_degree,
            render_mode="RGB+ED",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            # radius_clip=3,
        )

        if render_mode == "RGB":
            render = render_colors[0, ..., 0:3].cpu().numpy()
        elif render_mode == "ED":
            render = 1 / render_colors[0, ..., 3:].repeat_interleave(3, dim=-1).cpu().numpy()
        return render

class ViserViewer:
    def __init__(self, port):
        self.port = port
        self.server = viser.ViserServer(port=port)

        self.need_update = False

        with self.server.gui.add_folder("Playback"):
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.timestamp = self.server.add_slider(
                "Timestamp", min=0, max=49, step=1, initial_value=0
            )
            self.gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            self.gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)

            self.gui_show_depth = self.server.gui.add_checkbox(
                "Show Depth",
                initial_value=False,
            )
        
        @self.gui_playing.on_update
        def _(_) -> None:
            self.timestamp.disabled = self.gui_playing.value
            self.gui_next_frame.disabled = self.gui_playing.value
            self.gui_prev_frame.disabled = self.gui_playing.value

        @self.timestamp.on_update
        def _(_):
            self.need_update = True
        
        # Frame step buttons.
        @self.gui_next_frame.on_click
        def _(_) -> None:
            self.timestamp.value = (self.timestamp.value + 1) % 50

        @self.gui_prev_frame.on_click
        def _(_) -> None:
            self.timestamp.value = (self.timestamp.value - 1) % 50

        @self.gui_show_depth.on_update
        def _(_) -> None:
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True
        
        # self.scene_rep = DynGSRenderer(args)
    
    def set_scene_rep(self, scene_rep):
        self.scene_rep = scene_rep
    
    def render(self, camera, timestamp):
        if self.gui_show_depth.value:
            return self.scene_rep.render(camera, timestamp, "ED")
        else:
            return self.scene_rep.render(camera, timestamp, "RGB")
    
    @torch.no_grad()
    def update(self):
        if self.need_update:
            start = time.time()
            for client in self.server.get_clients().values():
                camera = client.camera
                timestamp = self.timestamp.value / 50
                w2c = get_w2c(camera)
                try:
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    out = self.render(camera, timestamp)

                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda)/1000.

                except RuntimeError as e:
                    print(e)
                    interval = 1
                    continue

                client.set_background_image(out, format="jpeg")

            # self.need_update = False

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    dyn_gs = DynGSRenderer(args)

    gui = ViserViewer(port=8080)

    gui.set_scene_rep(dyn_gs)

    while(True):
        if gui.gui_playing.value:
            gui.timestamp.value = (gui.timestamp.value + 1) % 50
        gui.update()


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt results/garden/ckpts/ckpt_3499_rank0.pt results/garden/ckpts/ckpt_3499_rank1.pt \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, inria")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
