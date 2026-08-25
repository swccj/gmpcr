# GMPCR

Official implementation of **Spectral Consistency-Guided Multiview Point
Cloud Registration for Low-Overlap Scenes**.

This release contains the current SCG Rayleigh-confidence implementation and
the selected configurations for 3DMatch, 3DLoMatch, ScanNet, and ETH. Dataset
files, YOHO descriptors, paper sources, and third-party comparison repositories
are intentionally excluded.

## Method in this release

For each refined second-order compatibility graph
\(\widetilde{\mathbf B}_{ij}\), scan-pair confidence is computed using the
Rayleigh quotient

\[
\widehat{s}_{ij}
=
\widetilde{\mathbf v}_{ij}^{\mathsf T}
\widetilde{\mathbf B}_{ij}
\widetilde{\mathbf v}_{ij},
\qquad
\|\widetilde{\mathbf v}_{ij}\|_2=1.
\]

The unordered scores in a scene are L2-normalized before row-wise Top-K pair
selection. The selected pose-graph edges are initialized using inlier support
and, where enabled by the dataset configuration, scan-pair confidence. Global
poses are estimated with adaptive history-aware synchronization. 3DLoMatch
uses decoupled translation IRLS.

The recovery strength is loaded from the selected dataset configuration.
Setting `rho=0` disables recovery, as in the selected 3DMatch and 3DLoMatch
configurations.

## Repository layout

```text
GMPCR/
├── main.py                      # the only public entry point
├── gmpcr.py                     # current Rayleigh-confidence implementation
├── configs/                     # one selected config per dataset
├── dataops/                     # dataset loading
├── TransSync/                   # robust transformation synchronization
├── utils/                       # matching and evaluation utilities
└── data/README.md               # expected dataset layout
```

## Installation

The validated environment uses Python 3.11, PyTorch 2.11.0 with CUDA 12.8,
NumPy 2.4.5, Open3D 0.19.0, and python-igraph 1.0.0.

Using Conda:

```bash
conda env create -f environment.yml
conda activate gmpcr
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the virtual environment with
`.venv\Scripts\Activate.ps1`. If CUDA 12.8 is unavailable, install the
appropriate PyTorch build first and then install the remaining dependencies.
The code also runs on CPU, although full-dataset evaluation is considerably
slower.

## Data preparation

Prepare the benchmark point clouds, keypoint indices, ground-truth logs, and
YOHO descriptors according to [data/README.md](data/README.md). Data are not
tracked by Git.

You can keep the data outside the repository and pass its location explicitly:

```bash
python main.py --dataset 3dmatch --data-dir /path/to/gmpcr_data
```

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

With seed 2026, the current full-dataset validation obtained registration
recalls of **97.2%** on 3DMatch and **89.6%** on 3DLoMatch.

## Reproducibility notes

- Random seeds are applied to Python, NumPy, PyTorch, and CUDA.
- PyTorch deterministic algorithms are requested where available.
- Dataset-specific parameters are stored only in the four JSON files under
  `configs/`.
- Generated results, datasets, model weights, and cached keypoints are ignored
  by `.gitignore`.

## Citation

Please cite the associated GMPCR paper if this code is useful in your work.
Citation metadata are provided in `CITATION.cff`.
