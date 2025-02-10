# GSCodec Studio

GSCodec Studio is an open-source framework for Gaussian Splats Compression, including static and dynamic splats representation, reconstruction and compression. It is bulit upon an open-source 3D Gaussian Splatting library [gsplat](https://github.com/nerfstudio-project/gsplat), and extended to support 1) dynamic splats representation, 2) training-time compression simulation, 3) more test-time compression strategies.

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

Then, you can install the gsplat library extended with more compression features from source code. In this way it will build the CUDA code during installation.

```bash
pip install .
# pip install -e . (develop mode)
```

## Evaluation

**Preparations**
Same as gsplat, we need to install some extra dependencies and download the relevant datasets before the evaluation.

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# place other dataset under 'data' folder
```

**Static Gaussian Splats Compression**

We provide a script that enables more memory-efficient Gaussian splats while maintaining high visual quality, such as representing the Truck scene with only about 8MB of storage. 

```bash
bash benchmarks/compression/final_exp/mcmc_tt_sim.sh
```

## More examples
**Dynamic Gaussian Splats Compression**

**Extract Per-Frame Static Gaussian from Dynamic Splats**
