# GSCodec Studio

GSCodec Studio is an open-source framework for Gaussian Splats Compression, including static and dynamic splats representation, reconstruction and compression. It is bulit upon an open-source 3D Gaussian Splatting library [gsplat](https://github.com/nerfstudio-project/gsplat), and extended to support 1) dynamic splats representation, 2) training-time compression simulation, 3) more test-time compression strategies.

![Teaser](./assets/Teaser.png)

## Installation
### Repo. & Environment
```bash
# Clone the repo.
git clone https://github.com/JasonLSC/GSCodec_Studio.git --recursive
cd GSCodec_Studio

# Make a conda environment
conda create --name gscodec_studio python=3.12
conda activate gscodec_studio
```

### Packages Installation

Please install [Pytorch](https://pytorch.org/get-started/locally/) first. Then, you can install the gsplat library extended with more compression features from source code. In this way it will build the CUDA code during installation.

```bash
pip install .
```

If you want to do further development based on this framework, you use following command to install Python packages in editable mode.
```bash
pip install -e . # (develop)
```

## Examples

**Preparations**

Same as gsplat, we need to install some extra dependencies and download the relevant datasets before the evaluation.

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
```
You can download the Tanks and Temples dataset and Deep Blending dataset used in original 3DGS via [this link](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) and place these datasets under 'data' folder.
```bash
# place other dataset, e.g Tanks and Temples dataset, under 'data' folder
ln -s data/tandt /xxxx/Dataset/tandt
```

We also use third-party library, 'python-fpnge', to accelerate image saving operations during the experiment for now. We also use third-party library, 'gridencoder', to facilitate hash encoding.

```bash
cd ..
pip install third_party/python-fpnge-master
pip install third_party/gridencoder
```

Before we start running scripts, we also need to install library for [vector quantization](https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) and [plas sorting](https://github.com/fraunhoferhhi/PLAS).
```bash
# refer to https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install to see how to install TorchPQ
pip install git+https://github.com/fraunhoferhhi/PLAS.git
```

### Static Gaussian Splats Training and Compression

We provide a script that enables more memory-efficient Gaussian splats while maintaining high visual quality, such as representing the Truck scene with only about 8MB of storage. The script includes 1) the static splats training with compression simulation, 2) the compression of trained static splats, and 3) the metric evaluation of uncompressed and compressed static splats.

```bash
# Tanks and Temples dataset
bash benchmarks/compression/final_exp/mcmc_tt_sim.sh
```

### Dynamic Gaussian Splats Training and Compression

First, please follow the dataset preprocessing instruction described in [this file](mpeg_gsc_utils/multiview_video_preprocess/README.md) for training data prepration.

Next, run the script for dynamic gaussian splats training and compression.
```bash
cd examples
bash benchmarks/dyngs/dyngs.sh
```

### Extract Per-Frame Static Gaussian from Dynamic Splats

If you finsh the training of dynamic splats, then you can use the script to extract static gaussian splats stored at discrte timesteps in ".ply" file
```bash
cd examples
bash benchmarks/dyngs/export_plys.sh
```

### Load .ply File and Render

If you want to directly read and render splats exported as PLY files, you can use the following script.
Note: You need to modify the PLY file path in the scripts.
```bash
cd examples
bash benchmarks/load_ply_and_render.sh
```

### Compress Tracked Gaussian Splats Sequences via Video Codec

If you're interested in MPEG Gaussian Splats Coding, we have implemented a simple coding method for compressing temporally tracked Gaussian Splats using Video Codec as the core component. You can try this method using the script below.
```bash
cd examples
bash benchmarks/mpeg/video_anchor_bench.sh
```

To compare with the Point Cloud Compression-based approach in MPEG Gaussian Splats Coding group, we have developed a simple wrapper based on their software. This allows us to evaluate both methods using the same evaluation protocol. You can try the Point Cloud Compression-based approach using the script below.
```bash
cd examples
bash benchmarks/mpeg/pcc_anchor_bench.sh
```

**Note**: 
Before using the scripts mentioned above, please note the following:
1. Please modify the input parameters ``--ply_dir``, ``--data_dir``, and ``--result_dir`` in the script to match your local paths.
2. These experiments involve third-party programs, including point cloud codec and QMIV (quality evaluation software). If you need newer versions, you can compile them yourself and replace the current executables under ``examples/helper``.

## Contributors

This project is developed by the following contributors:

- Sicheng Li: jasonlisicheng@zju.edu.cn
- Chengzhen Wu: chengzhenwu@zju.edu.cn

If you have any questions about this project, please feel free to contact us.

## Acknowledgement
This project is bulit on [gsplat](https://github.com/nerfstudio-project/gsplat). We thank all contributors from [gsplat](https://github.com/nerfstudio-project/gsplat) for building such a great open-source project.