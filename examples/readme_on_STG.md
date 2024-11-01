## 代码运行说明
1. 可能需要在gsplat的根目录下运行setup.py来安装STG Strategy
2. 运行前，需要在simple trainer STG.py一开始的config里设置model path，data dir与’result dir。data dir是N3D数据集的路径，需要按照STG的步骤下载并对数据集进行预处理。
3. 运行python simple trainer STG.py即可

P.S 假如运行duration=20的话，可能运行到20000iters左右会卡住，这个不是GPU存储的问题，有可能是CPU里存了太多东西?可以的话，运行7000iters一般就足够收敛了，我这边又跑了一次7000iters psnr能到33了。

## Sicheng Steps
1. clone env
``` python
conda create -n stg_compress --clone gsplat
```
maybe takes almost 10 mins

2. install modified gsplat backend
```
pip install -e . 
```
3. preprocess of video datasets (https://github.com/oppo-us-research/SpacetimeGaussians?tab=readme-ov-file#processing-datasets)
run "python script/pre_n3d.py" on 6 scenes in neu_3d_video 

   - 18 mins to convert mp4 to png
   - 12 mins to convert dynerf to colmap
   
   Note: 实际上运行上述程序，只会做50 frames的SfM...;所以后续还得再抽时间做第50-300帧的预处理

   State: Coffee_martini已做



