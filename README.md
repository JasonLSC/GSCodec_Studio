# GSCodec Studio

GSCodec Studio is an open-source framework for Gaussian Splats Compression, including static and dynamic splats representation, reconstruction and compression. It is bulit upon an open-source 3D Gaussian Splatting library [gsplat](https://github.com/nerfstudio-project/gsplat), and extended to support 1) dynamic splats representation, 2) training-time compression simulation, 3) more test-time compression strategies.

![Teaser](./assets/Teaser.png)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

Then, you can install the gsplat library extended with more compression features from source code. In this way it will build the CUDA code during installation.

```bash
pip install .
# pip install -e . (develop mode)
```

## Examples

**Preparations**
Same as gsplat, we need to install some extra dependencies and download the relevant datasets before the evaluation.

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# or place other dataset under 'data' folder
ln -s data/tandt /xxxx/Dataset/tandt
```

We also use third-party library, 'python-fpnge', to accelerate image saving operations during the experiment for now. 

```bash
cd ../third_party/python-fpnge-master
pip install .
```

**Static Gaussian Splats Training and Compression**

We provide a script that enables more memory-efficient Gaussian splats while maintaining high visual quality, such as representing the Truck scene with only about 8MB of storage. The script includes 1) the static splats training with compression simulation, 2) the compression of trained static splats, and 3) the metric evaluation of uncompressed and compressed static splats.

```bash
bash benchmarks/compression/final_exp/mcmc_tt_sim.sh
```

**Dynamic Gaussian Splats Training and Compression**
(Will be provided on 2/15/2025)

**Extract Per-Frame Static Gaussian from Dynamic Splats**
(Will be provided on 2/15/2025)

## Contributors

This project is developed by the following contributors:

- Sicheng Li: jasonlisicheng@zju.edu.cn
- Chengzhen Wu: chengzhenwu@zju.edu.cn

If you have any questions about this project, please feel free to contact us.

## Acknowledgement
This project is bulit on [gsplat](https://github.com/nerfstudio-project/gsplat). We thank all contributors from [gsplat](https://github.com/nerfstudio-project/gsplat) for building such a great open-source project.