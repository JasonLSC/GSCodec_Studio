# Designed for preprocessed N3D Dataset, preprocess follows STG's method

import os
import random
import json
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio

import numpy as np
import torch
from pycolmap import SceneManager

from helper.STG.dataset_readers import sceneLoadTypeCallbacks
from helper.STG.camera_utils import camera_to_JSON, cameraList_from_camInfosv2
from helper.STG.general_utils import PILtoTorch

# reference to STG's scene __init__.py
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
        resolution: int = 2,
        data_device: str = "cuda",
    ):
        self.model_path = model_path 
        self.source_path = source_path 
        self.images_phrase = images_phrase
        self.eval = eval
        self.duration = duration
        self.resolution_scales = resolution_scales
        
        self.train_cameras = {}
        self.test_cameras = {}
        raydict = {}
        
        if loader == "colmap": # colmapvalid only for testing
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.source_path, self.images_phrase, self.eval, multiview, duration=self.duration)
        else:
            assert False, "Could not recognize scene type!"

        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file, indent=2)
            
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # need modification
        class ModelParams(): 
            def __init__(self):
                self.resolution = resolution
                self.data_device = data_device
        args = ModelParams()
        self.args = args

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")  
            self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args)
        
        for cam in self.train_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                # rays_o, rays_d = 1, cameradirect
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W
        
        for cam in self.test_cameras[resolution_scale]:
            if cam.image_name not in raydict and cam.rayo is not None:
                raydict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W

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
     
                
class Dataset:
    def __init__(
        self,
        parser: Parser,
        split: str = "train",

    ): 
        self.parser = parser
        self.resolution_scale = self.parser.resolution_scales[0]
        if split == "train":
            self.scene_info = self.parser.scene_info[1]
            self.cam_list = self.parser.train_cameras[self.resolution_scale]
            self.camtoworld = self.parser.camtoworld
        elif split == "test":
            self.scene_info = self.parser.scene_info[2]
            self.cam_list = self.parser.test_cameras[self.resolution_scale]
            self.camtoworld = self.parser.camtoworld_test
        else:
            assert False, "Invalid split input!"
        for i in range(len(self.cam_list)):
            self.cam_list[i].rays = self.cam_list[i].rays.to("cpu")
        
    def __len__(self):
        return len(self.scene_info)
        
    def __getitem__(self, item: int) -> Dict[str, Any]:
        K = self.parser.K
        scale = self.parser.args.resolution
        resolution = (int(self.scene_info[item].width / scale), int(self.scene_info[item].height / scale))
        image = PILtoTorch(self.scene_info[item].image, resolution).permute(1,2,0)
        camtoworld = self.camtoworld

        data = {
            "K": torch.from_numpy(K), 
            "R": torch.from_numpy(self.scene_info[item].R), 
            "T": torch.from_numpy(self.scene_info[item].T),
            "image": image,  
            "timestamp": self.scene_info[item].timestamp,
            "ray": self.cam_list[item].rays,
            "camtoworld": torch.from_numpy(camtoworld[item]),
        }            
        return data

# to test this dataset loader, run: python INVR_N3D.py
if __name__ == "__main__":
    model_path = "/home/czwu/gsplat_output/STG_N3D_version"
    source_path = "/data/czwu/Neural_3D_Dataset/flame_steak/colmap_0"

    parser = Parser(model_path=model_path, source_path=source_path)
    dataset = Dataset(parser=parser, split="train")
    
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainloader_iter = iter(trainloader)
    for i in range(50):
        data = next(trainloader_iter)
        print(data["timestamp"])
        
    # data = next(trainloader_iter)
    import pdb; pdb.set_trace()