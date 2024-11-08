# 1. 因为ours静态的参数没有填，所以与其发运行的latex格式的结果，我还是直接把这几个dict发给你
# 2. 这里的数据都是参考各自对应论文上的数据
# 3. HAC, IGS与SOG论文中都基本没有FPS的结果，即便有，也是针对只针对某个数据集（HAC对Synthetic_NeRF）或者某个特定场景(SOG, Truck)
results_Tanks_and_Temples = {
    'HAC':{
        'HAC-lowrate': [24.04, 0.846, 0.187, 8.10, None],
        'HAC-highrate': [24.40, 0.853, 0.177, 11.24, None],
    },
    'IGS':{
        'IGS-lowrate': [23.70, 0.836, 0.227, 8.44, None],
        'IGS-highrate': [24.05, 0.849, 0.210, 12.5, None],
    },
    'SOG':{
        'SOG-w/o SH': [23.15, 0.828, 0.198, 9.3, None], # 论文上的数字有点奇怪，最后我还是选择用了3DGS.zip里的结果
        'SOG': [23.56, 0.837, 0.186, 22.8, None],
    },
    'gsplat_comp.':{
        'gsplat-1M-comp.': [23.97, 0.854, 0.164, 15.32, None]
    },
    'Ours':{
        'Ours-lowrate': [],
        'Ours-highrate': []
    }
}
results_MipNeRF_360 = {
    'HAC':{
        'HAC-lowrate': [27.53, 0.807, 0.238, 15.26, None],
        'HAC-highrate': [27.77, 0.811, 0.230, 21.87, None]
    },
    'IGS':{
        'IGS-lowrate': [27.33, 0.809, 0.257, 12.5, None],
        'IGS-highrate': [27.62, 0.819, 0.247, 25.4, None],
    },
    'SOG':{
        'SOG-w/o SH': [27.02, 0.803, 0.232, 16.7, None],
        'SOG': [27.64, 0.814, 0.220, 40.3, None],
    },
    'gsplat_comp':{
        'gsplat_comp': [27.24, 0.808, 0.231, 14.82, None],
    },
    'Ours':{
        'Ours-lowrate': [],
        'Ours-highrate': []
    }
}
results_Deep_Blending = {
    'HAC':{
        'HAC-lowrate': [29.98, 0.902, 0.269, 4.35, None],
        'HAC-highrate': [30.34, 0.906, 0.258, 6.35, None],
    },
    'IGS':{
        'IGS-lowrate': [30.63, 0.904, 0.293, 6.34, None],
        'IGS-highrate': [32.33, 0.924, 0.253, 7.74, None],
    },
    'SOG':{
        'SOG-w/o SH': [30.50, 0.908, 0.261, 5.5, None],
        'SOG': [30.35, 0.909, 0.258, 16.8, None],
    },
    'gsplat_comp':{
        'gsplat_comp': [27.24, 0.808, 0.231, 14.82, None],
    },
    'Ours':{
        'Ours-lowrate': [],
        'Ours-highrate': []
    }
}


# results_Synthetic_NeRF = {
#     'HAC-lowrate': [33.24, 0.967, 0.037, 1.18, None], # FPS 341 HAC论文里4.5提到HAC方法在Synthetic-NeRF数据集上FPS是341，不过我不确定high和low bit rate是不是应该都是这个FPS
#     'HAC-highrate': [33.71, 0.968, 0.034, 1.86, None], # FPS 341
#     'IGS-lowrate': [33.36, 0.971, 0.036, 1.85, None],
#     'IGS-highrate': [34.18, 0.975, 0.032, 2.72, None],
#     'SOG-w/o SH': [31.75, 0.961, 0.040, 2.0, None],
#     'SOG': [33.70, 0.969,  0.031, 4.1, None],
#     'My beloved algorithm': [38.924123, 0.8977, 0.051241, 12.456, None]}