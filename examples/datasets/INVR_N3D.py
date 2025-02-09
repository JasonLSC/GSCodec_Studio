# Designed for preprocessed N3D Dataset, preprocess follows STG's method

import os
import random
import json
from typing import Any, Dict, List, Optional
import re
import cv2
import imageio.v2 as imageio
from PIL import Image
import numpy as np

import torch
from pycolmap import SceneManager
try:
    from helper.STG.dataset_readers import sceneLoadTypeCallbacks
    from helper.STG.camera_utils import camera_to_JSON, cameraList_from_camInfosv2
    from helper.STG.general_utils import PILtoTorch, PILtoTorch_new
    from helper.STG.time_utils import timer, timeblock
except:
    from examples.helper.STG.dataset_readers import sceneLoadTypeCallbacks
    from examples.helper.STG.camera_utils import camera_to_JSON, cameraList_from_camInfosv2
    from examples.helper.STG.general_utils import PILtoTorch, PILtoTorch_new
    from examples.helper.STG.time_utils import timer, timeblock

# reference to STG's scene __init__.py
@timer
class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        model_path: str,
        source_path: str,
        loader: str ="colmap",
        images_phrase: str ="images",
        shuffle: bool = True,
        eval: bool = False,
        multiview: bool = False,
        duration: int = 5, # only for testing
        resolution_scales: list = [1.0],
        downscale_factor: int = 2,
        data_device: str = "cpu",
        test_view_id: List[int] = [0]
    ):
        self.model_path = model_path 
        self.source_path = source_path 
        self.images_phrase = images_phrase
        self.eval = eval
        self.duration = duration
        self.resolution_scales = resolution_scales
        self.test_view_id = test_view_id
        
        self.train_cameras = {}
        self.test_cameras = {}
        raydict = {}
        
        # Get scene info
        if loader == "colmap": # colmapvalid only for testing
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.source_path, self.images_phrase, self.eval, multiview, duration=self.duration, test_view_id=self.test_view_id, downscale_factor=downscale_factor) # SceneInfo() - NamedTuple
        # elif loader == "invr":
        #     scene_info = sceneLoadTypeCallbacks["INVR"](self.source_path, self.images_phrase, self.eval, multiview, duration=self.duration) # SceneInfo() - NamedTuple
        else:
            assert False, "Could not recognize scene type!"

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # need modification
        class ModelParams(): 
            def __init__(self):
                self.downscale_factor = downscale_factor
                self.data_device = data_device
        args = ModelParams()
        self.args = args

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")  
            self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args) # Dist[float, List[Camera()]]
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args) # Dist[float, List[Camera()]]
        
        for cam in self.train_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                # rays_o, rays_d = 1, cameradirect
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1) # 1 x 6 x H x W
        
        for cam in self.test_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1) # 1 x 6 x H x W

        for cam in self.train_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name] # should be direct ?

        for cam in self.test_cameras[resolution_scale]:
            cam.rays = raydict[cam.image_name] # should be direct ?
        
        # self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args)
        self.points = scene_info.point_cloud.points
        self.points_rgb = scene_info.point_cloud.colors
        
        self.timestamp = scene_info.point_cloud.times
        self.scene_scale = self.cameras_extent # TODO Is this correct? check
        
        self.K = scene_info.train_cameras[int(resolution_scale)].K

        self.camtoworld = scene_info.nerf_normalization['camtoworld']
        self.camtoworld_test = scene_info.nerf_normalization_test['camtoworld']
        self.scene_info = scene_info # TODO may consume unnecessary storage, check
     
                
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        num_views: int = 1,
        use_fake_length: bool = False,
        fake_length: int = -1
    ): 
        self.use_fake_length = use_fake_length
        if self.use_fake_length:
           assert fake_length > -1
        self.fake_length = fake_length
        self.num_views = num_views
        self.parser = parser
        self.resolution_scale = self.parser.resolution_scales[0]
        self.split = split
        
        if split == "train":
            self.scene_info = self.parser.scene_info[1]
            self.cam_list = self.parser.train_cameras[self.resolution_scale] # actually, img_list, [[v0_fr0: v0_fr49], [v1_fr0: v1_fr49], ..., [v20_fr0: v20_fr49]]
            self.camtoworld = self.parser.camtoworld
            # scene_by_path = {i.image_path : i for i in self.scene_info}
            scene_by_t = dict()
            for i, cinfo in enumerate(self.scene_info):
                tid = int(re.search(r'colmap_(\d+)', cinfo.image_path).group(1))
                if not tid in scene_by_t:
                    scene_by_t[tid] = [(i, self.cam_list[i], cinfo)]
                else:
                    scene_by_t[tid] += [(i, self.cam_list[i], cinfo)]
            self.scene_by_t = scene_by_t
        elif split == "test":
            self.scene_info = self.parser.scene_info[2]
            self.cam_list = self.parser.test_cameras[self.resolution_scale]
            self.camtoworld = self.parser.camtoworld_test
            # scene_by_path = {i.image_path : i for i in self.scene_info}
            scene_by_t = dict()
            for i, cinfo in enumerate(self.scene_info):
                tid = int(re.search(r'colmap_(\d+)', cinfo.image_path).group(1))
                if not tid in scene_by_t:
                    scene_by_t[tid] = [(i, self.cam_list[i], cinfo)]
                else:
                    scene_by_t[tid] += [(i, self.cam_list[i], cinfo)]
            self.scene_by_t = scene_by_t
        else:
            assert False, "Invalid split input!"
        
        self.start_frame = min(scene_by_t.keys())
        
    def __len__(self): # num of timestamp
        return  self.fake_length if self.use_fake_length else len(self.scene_by_t)
        # return len(self.scene_info)

    def fetch_image(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
        
    def __getitem__(self, index: int) -> Dict[str, Any]:
        tid = index % len(self.scene_by_t)
        t_infos = self.scene_by_t[tid + self.start_frame]
        if self.split == "train":
            try:
                frame_infos = random.sample(t_infos, k=self.num_views)
            except: #replace
                frame_infos = random.choices(t_infos, k=self.num_views)
        else:
            frame_infos = t_infos[:self.num_views] # take out frames in single imgstamp by default order

        K = self.parser.K
        downscale_factor = self.parser.args.downscale_factor
        Ks, images, image_paths, rays, timesteps, camtoworlds = [], [], [], [], [], []
        for globalid, cami, finfo in frame_infos:
            resolution = (int(finfo.width / downscale_factor), int(finfo.height / downscale_factor))
            
            images.append(PILtoTorch_new(self.fetch_image(finfo.image_path), resolution).permute(1,2,0))
            image_paths.append(finfo.image_path)
            camtoworlds.append(torch.from_numpy(self.camtoworld[globalid]))
            timesteps.append(tid/len(self.scene_by_t))
            Ks.append(torch.from_numpy(self.parser.K))
            rays.append(cami.rays[0])

        data = {
            "K": torch.stack(Ks, dim=0).to(torch.float32), 
            "image": torch.stack(images, dim = 0).to(torch.float32),  
            "image_path":image_paths,
            "timestamp": timesteps,
            "ray":  torch.stack(rays, dim = 0).to(torch.float32),
            "camtoworld": torch.stack(camtoworlds, dim = 0).to(torch.float32),
        }            
        return data





# class CustomCollate:
#     def __init__(self, pad_value: float = 0.0):
#         self.pad_value = pad_value
        
#     def __call__(self, batch: List[Dict]) -> Dict:
#         # 获取batch中所有键
#         keys = batch[0].keys()
        
#         collated_batch = {}
#         for key in keys:
#             pass
#             # if key == "image":
#             #     images = [item[key] for item in batch]
#             #     collated_batch[key] = torch.stack(images)
                
#             # elif key == "label":
#             #     labels = [item[key] for item in batch]
#             #     collated_batch[key] = torch.tensor(labels)
                
#             # elif key == "mask":
#             #     masks = [item[key] for item in batch]
#             #     collated_batch[key] = torch.stack(masks)
                
#         return collated_batch


# to test this dataset loader, run: python INVR_N3D.py
if __name__ == "__main__":
    model_path = "examples/results/stg_neu3d"
    source_path = "examples/data/neural_3d/flame_steak/colmap_0"

    parser = Parser(model_path=model_path, source_path=source_path)
    dataset = Dataset(parser=parser, split="train", num_views = 2)
    # collate_fn = CustomCollate(pad_value=0.0)

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        # persistent_workers=True,
        pin_memory=True,
        # collate_fn=collate_fn
    )
    trainloader_iter = iter(trainloader)
    for i, batch in enumerate(trainloader_iter):

        # data = next(trainloader_iter)
        print(i)
        
    # data = next(trainloader_iter)
    # import pdb; pdb.set_trace()