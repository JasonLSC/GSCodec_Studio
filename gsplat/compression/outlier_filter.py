from typing import Dict, final

import numpy as np
from scipy.spatial import KDTree
import torch
from torch import Tensor

def filter_splats(splats: Dict[str, Tensor], opa_thres: float=0.005, std_factor: float=2.0, k_neighbors: int=10):
    # 1. 首先基于不透明度过滤
    opacity_mask = torch.sigmoid(splats["opacities"]) >= opa_thres

    # # 2. 使用KD树计算每个点到其k个最近邻的平均距离
    # pos = splats["means"].cpu().numpy()
    # kdtree = KDTree(pos)
    # distances, _ = kdtree.query(pos, k=k_neighbors)
    # mean_distances = np.mean(distances, axis=1)
    
    # # 3. 计算平均距离的统计特征
    # dist_mean = np.mean(mean_distances)
    # dist_std = np.std(mean_distances)
    
    # # 4. 基于距离判定离群点
    # distance_mask = mean_distances <= (dist_mean + std_factor * dist_std)
    # distance_mask = torch.from_numpy(distance_mask).to(opacity_mask.device)
    
    # # 5. 组合两个过滤条件
    # ## v1: 保守方案
    # # outlier = torch.logical_and(~opacity_mask, ~distance_mask) # 既在位置上离群，且不透明度又低
    # # valid_mask = ~outlier
    # ## v2: 激进方案
    # # valid_mask = torch.logical_and(opacity_mask, distance_mask)
    valid_mask = opacity_mask

    # 6. 逐一过滤元素
    for n, v in splats.items():
        splats[n] = v[valid_mask]
    
    return valid_mask, splats