from typing import Dict, final

import numpy as np
from scipy.spatial import KDTree
import torch
from torch import Tensor

def filter_splats(splats: Dict[str, Tensor], opa_thres: float=0.005, std_factor: float=2.0, k_neighbors: int=10):
    opacity_mask = torch.sigmoid(splats["opacities"]) >= opa_thres

    
    # pos = splats["means"].cpu().numpy()
    # kdtree = KDTree(pos)
    # distances, _ = kdtree.query(pos, k=k_neighbors)
    # mean_distances = np.mean(distances, axis=1)
    
    #
    # dist_mean = np.mean(mean_distances)
    # dist_std = np.std(mean_distances)
    
    # distance_mask = mean_distances <= (dist_mean + std_factor * dist_std)
    # distance_mask = torch.from_numpy(distance_mask).to(opacity_mask.device)
    
    # ## v1: 
    # # outlier = torch.logical_and(~opacity_mask, ~distance_mask) # 既在位置上离群，且不透明度又低
    # # valid_mask = ~outlier
    # ## v2:
    # # valid_mask = torch.logical_and(opacity_mask, distance_mask)
    valid_mask = opacity_mask

    for n, v in splats.items():
        splats[n] = v[valid_mask]
    
    return valid_mask, splats