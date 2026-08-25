# GMPCR

这是论文 **Spectral Consistency-Guided Multiview Point Cloud Registration
for Low-Overlap Scenes** 的公开实现整理版。

该目录只保留当前采用的 SCG Rayleigh 置信度方法、四个数据集的最优配置和
运行所需代码。数据集、YOHO 描述子、论文源文件和第三方对比方法均未包含在
发布目录中。

## 安装

推荐使用已验证的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate gmpcr
```

已验证环境为 Python 3.11、PyTorch 2.11.0 + CUDA 12.8、NumPy 2.4.5、
Open3D 0.19.0 和 python-igraph 1.0.0。如服务器 CUDA 版本不同，请先安装
对应版本的 PyTorch，再安装其余依赖。

## 数据

按照 [data/README.md](data/README.md) 准备点云、关键点索引、GT 日志和 YOHO
描述子。数据可以位于项目目录之外，通过 `--data-dir` 指定：

```bash
python main.py --dataset 3dmatch --data-dir /path/to/gmpcr_data
```

## 运行

发布版只保留 `main.py` 作为入口，程序会按照数据集名称自动加载对应的最优参数：

```bash
python main.py \
  --dataset 3dLomatch \
  --data-dir /path/to/gmpcr_data \
  --save_dir ./pre/3dLomatch
```

支持的数据集名称为 `3dmatch`、`3dLomatch`、`scannet` 和 `ETH`。程序会自动
读取 `configs/` 中对应的最优配置。

## 当前配置说明

恢复强度由数据集配置自动加载；3DMatch 和 3DLoMatch 的当前最优配置均设置
`rho=0`，即不启用恢复项。固定随机种子 2026 时，当前全量验证结果为：

- 3DMatch RR：97.1991%；
- 3DLoMatch RR：89.6363%。

如使用本代码，请引用对应的 GMPCR 论文；作者信息见 `CITATION.cff`。
