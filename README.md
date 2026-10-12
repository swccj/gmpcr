# GMPCR

Official implementation of **Spectral Consistency-Guided Multiview Point
Cloud Registration for Low-Overlap Scenes**.

## Installation

The validated environment uses Python 3.11, PyTorch 2.11.0 with CUDA 12.8,
NumPy 2.4.5, Open3D 0.19.0, and python-igraph 1.0.0.

Using Conda:

```bash
conda env create -f environment.yml
conda activate gmpcr
```


## Data preparation

Prepare the benchmark point clouds, keypoint indices, ground-truth logs, and
YOHO descriptors according to [data/README.md](data/README.md).

## Evaluation

Run one dataset with its selected configuration:

```bash
python main.py \
  --dataset 3dmatch \
  --data-dir /path/to/gmpcr_data \
  --save_dir ./pre/3dmatch
```

Supported dataset names are `3dmatch`, `3dLomatch`, `scannet`, and `ETH`.
The entry point automatically loads the corresponding file from `configs/`.
To evaluate a single scene, add `--scene <scene-name>`.

Predicted poses and evaluation logs are written beneath `--save_dir`.

## Selected configurations

| Dataset | Configuration |
|---|---|
| 3DMatch | `configs/3dmatch.json` |
| 3DLoMatch | `configs/3dlomatch.json` |
| ScanNet | `configs/scannet.json` |
| ETH | `configs/eth.json` |


## Citation

Please cite the associated paper if this code is useful in your work.
