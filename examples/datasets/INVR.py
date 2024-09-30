import os
import json
import numpy as np
import torch
from tqdm import tqdm
import imageio.v2 as imageio
from multiprocessing.pool import ThreadPool
from typing import Dict, Any

class Parser:
    """INVR parser."""
    
    def __init__(
        self,
        data_dir: str,
        extension: str = ".png",
        set: str = "train",
        # Anymore parameters?
    ):
        self.data_dir = data_dir
        self.extension = extension
        self.set = set
        
        # Read json file according to data_dir 
        if os.path.exists(os.path.join(self.data_dir, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
        else:
            assert False, "Could not recognize scene type!"
        
        if set == "train":
            # Read Training Dataset
            with open(os.path.join(self.data_dir, "transforms_train.json")) as json_file:
                contents = json.load(json_file) 
        elif set == "test":
            with open(os.path.join(self.data_dir, "transforms_test.json")) as json_file:
                contents = json.load(json_file) 
        else:
            assert False, "Could not recognize set type!"
            
        # Assuiming Intrinsics of all cameras are the same, which may not always be true
        self.width = contents["w"]
        self.height = contents["h"]
        self.fl_x = contents["fl_x"]
        self.fl_y = contents["fl_y"]
        self.cx = contents["cx"]
        self.cy = contents["cy"]
        self.K = torch.zeros(3,3)
        self.K[0, 0] = self.fl_x; self.K[1, 1] = self.fl_y; self.K[0, 2] = self.cx; self.K[1, 2] = self.cy; self.K[2, 2] = 1
        
        self.image_id = []; self.timestamp = []; self.c2w=[]; self.R = []; self.T = []; self.image = []
        
        # Read frames
        frames = contents["frames"]
        tbar = tqdm(range(len(frames)))
        def frame_read_fn(idx_frame):
            idx = idx_frame[0] # equals to image id?
            frame = idx_frame[1]
            timestamp = frame.get('time', 0.0)
            cam_name = os.path.join(self.data_dir, frame.get('file_path', f'{idx:04d}') + extension)
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            self.image_id.append(idx)
            self.timestamp.append(timestamp)
            self.c2w.append(c2w)
            self.R.append(R)
            self.T.append(T)
            self.image.append(imageio.imread(cam_name))
            
            tbar.update(1)
        
        with ThreadPool() as pool:
            pool.map(frame_read_fn, zip(list(range(len(frames))), frames))
            pool.close()
            pool.join()
            
        
class Dataset:
    def __init__(
        self,
        parser: Parser,
        # Anymore params?
    ): 
        self.parser = parser
        
    def __len__(self):
        return len(self.parser.image_id)
        
    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = {
            "K": self.parser.K.float(),
            "c2w": self.parser.c2w[item],
            # "R": self.parser.R[item],
            # "T": self.parser.T[item],
            "image": torch.from_numpy(self.parser.image[item]).float(),
            "timestamp": self.parser.timestamp[item],
            "image_id": item,
        }            
        return data
        
  
if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/czwu/Bartender")
    args = parser.parse_args()
    
    parser_1 = Parser(data_dir=args.data_dir)
    dataset = Dataset(parser_1)
    
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainloader_iter = iter(trainloader)
    data = next(trainloader_iter)
    import pdb; pdb.set_trace()