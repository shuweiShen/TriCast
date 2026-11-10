# TriCast

**TriCast: Tri-view Time Series Forecasting with Frozen Language Backbones**

[English](#english) | [中文](#中文)

---

## English

### Overview

TriCast is a research implementation of time series forecasting that couples three complementary views of a look-back window — **time-domain patches**, **change-point statistics**, and **spectral features** — and feeds the fused representation into a **frozen pretrained language model**. Only lightweight adapters are trained; the LLM backbone stays intact.

The three views are aligned by dedicated projection blocks, fused by FiLM, refined by Fourier cross-attention, and routed to a set of experts conditioned on the change-point and spectral priors (CPRouter). The pipeline is configurable in six cumulative stages (`tricast_stage` 0–5), which makes module-wise ablation straightforward.

### Highlights

- **Frozen LLM backbone**: GPT-2 family, Qwen2.5 and TinyLlama are supported through a local registry; no backbone weights are saved in checkpoints.
- **Three views**: non-overlapping time-domain patches, multiscale distribution-shift priors (MSDR), and orthogonal rFFT spectral features (phase preserved).
- **Fusion stack**: three independent `AlignBlock`s, FiLM modulation, Fourier cross-attention, and a change-point-aware expert router.
- **Staged ablation**: `tricast_stage` 0–5 progressively enables MSDR, spectral alignment, FiLM, Fourier cross-attention and the expert router.
- **Reproducible experiments**: 7 backbones × 4 ETT datasets generated configs, plus batch runners and per-experiment PowerShell wrappers.

### Repository structure

```text
TriCast/
├── run_tricast.py                 # training / evaluation / smoke entry point
├── models/TriCast.py              # full model assembly
├── layers/TriCast.py              # MSDR, PatchEncoder, SpectralEncoder, AlignBlock, FiLM, FourierCrossAttn, CPRouter
├── data_provider/                 # ETT and custom CSV loaders
├── utils/
│   ├── llm_backbone.py            # local backbone loading and identity checks
│   └── timefeatures.py            # time-feature encoding used by the loaders
├── configs/
│   ├── llm_backbones.json         # backbone registry (model id, local path, hidden size, layers)
│   └── experiments/               # 28 generated experiment configs
├── scripts/
│   ├── download_tricast_llm.py    # explicit backbone download helper
│   ├── generate_tricast_experiments.py
│   ├── run_tricast_matrix.py      # batch runner (dry-run supported)
│   ├── TriCast_stages.ps1         # stage 0–5 sanity checks (Windows)
│   └── tricast/                   # 56 generated PowerShell train/test wrappers
├── tests/                         # module tests and multi-backbone tests
├── TriCast_IMPLEMENTATION.md      # implementation notes and staged design (中文)
├── TriCast_MULTILLM.md            # multi-backbone and batch usage guide (中文)
├── requirements.txt
└── LICENSE
```

Datasets, model weights, checkpoints and run logs are **not** tracked in this repository.

### Requirements

- Python 3.11
- PyTorch, Transformers (≥ 4.37 for Qwen2 support), NumPy, pandas, scikit-learn, scipy

```bash
pip install -r requirements.txt
```

### 1. Prepare datasets

Place the ETT CSVs under `dataset/ETT-small/`:

```text
dataset/ETT-small/ETTh1.csv
dataset/ETT-small/ETTh2.csv
dataset/ETT-small/ETTm1.csv
dataset/ETT-small/ETTm2.csv
```

The ETT datasets are public; download them from [Time-Series-Library](https://github.com/thuml/Time-Series-Library). The loaders reuse the original train/validation/test split and train-set standardization.

### 2. Prepare LLM backbones

Backbones are loaded **only from local paths** — nothing is downloaded automatically. Edit `configs/llm_backbones.json` and set `local_path` for the backbone you need, or use the download helper:

```bash
python scripts/download_tricast_llm.py --model gpt2 --output ./llm_weights/gpt2 --register
```

Supported registry entries: `gpt2`, `gpt2-large`, `gpt2-xl`, `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `tinyllama-1.1b`.

### 3. Quick start

Run a single experiment with the GPT-2 configuration on ETTh1:

```bash
# Check backbone and data splits without loading the full model
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode inspect

# Forward/backward smoke test
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode smoke --batch_size 1

# Train (best adapter is saved by validation MSE)
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode train

# Evaluate a saved checkpoint
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode eval \
  --checkpoint ./checkpoints/tricast_single/gpt2/ETTh1/best.pt \
  --output_dir ./logs/tricast_eval
```

Outputs (`best.pt`, `history.json`, `metrics.json`, `resolved_config.json`) are written to `output_dir`.

### 4. Stage-wise ablation

`--tricast_stage` enables modules cumulatively:

| Stage | Enabled components |
| --- | --- |
| 0 | Time-domain patch encoding → projection → frozen LLM → forecasting head |
| 1 | + MSDR change-point branch (prior generation only) |
| 2 | + spectral encoding and three AlignBlocks |
| 3 | + FiLM injection of the change-point prior |
| 4 | + dynamic spectral gating and Fourier cross-attention |
| 5 | + CPRouter and four experts (full model) |

```bash
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode smoke --tricast_stage 3
```

On Windows, the full 0–5 check sequence is available as:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/TriCast_stages.ps1
```

### 5. Batch experiments

```bash
# Preview every command without running it
python scripts/run_tricast_matrix.py --mode train-test --dry_run

# One backbone on selected horizons
python scripts/run_tricast_matrix.py --mode train --models gpt2-xl --pred_lens 96 192

# All 7 backbones × 4 datasets × 4 horizons, sequentially
python scripts/run_tricast_matrix.py --mode train-test
```

Per-experiment PowerShell wrappers (Windows) live in `scripts/tricast/`:

```powershell
./scripts/tricast/gpt2-xl_ETTh1_train.ps1 -DryRun
./scripts/tricast/gpt2-xl_ETTh1_train.ps1
```

The generated configs and wrappers can be recreated with:

```bash
python scripts/generate_tricast_experiments.py
```

> This overwrites all 28 configs and 56 wrappers, so do not run it after customizing them.

### 6. Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

- `tests/test_tricast.py` — module behavior of `MSDR`, `FiLM`, `FourierCrossAttn`, `CPRouter`, `PatchEncoder`.
- `tests/test_multillm.py` — local tiny GPT-2 / Qwen2 / Llama fixtures, dtype handling, truncation and identity checks.

### Notes

- Checkpoints store only trainable adapters plus the resolved config; frozen backbone weights are reloaded from the registry.
- `metrics.json` reports MSE/MAE on the scale of the original loader's train-set standardization, including output de-normalization.
- The numbers produced by small debug runs are for pipeline verification only and are not paper results.

### Acknowledgement

TriCast builds on ideas and infrastructure from [Time-LLM](https://github.com/KimMeen/Time-LLM) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library). The original baseline code is not included in this repository.

### License

[Apache License 2.0](LICENSE)

---

## 中文

### 简介

TriCast 是一个时间序列预测的研究实现。它把回看窗口的三种互补表示——**时域分块**、**突变点统计**、**频域特征**——对齐融合后送入**冻结的预训练语言模型**，只训练轻量适配器，主干权重保持不变。

三种表示分别经过独立的投影模块，由 FiLM 融合、Fourier 交叉注意力细化，并在突变与频谱先验的条件下通过 CPRouter 路由到多个专家。模型按六个累积阶段（`tricast_stage` 0–5）配置，便于逐模块消融。

### 主要特点

- **冻结 LLM 主干**：支持 GPT-2 系列、Qwen2.5 与 TinyLlama，通过本地注册表加载；检查点不保存主干权重。
- **三视图输入**：非重叠时域分块、多尺度分布突变先验（MSDR）、正交归一化 rFFT 频谱（保留相位）。
- **融合结构**：三个独立 `AlignBlock`、FiLM 调制、Fourier 交叉注意力、突变感知专家路由。
- **分阶段消融**：`tricast_stage` 0–5 依次启用 MSDR、频谱对齐、FiLM、Fourier 交叉注意力与专家路由。
- **可复现实验**：7 种主干 × 4 个 ETT 数据集的生成式配置，配套批量运行器与 PowerShell 脚本。

### 仓库结构

```text
TriCast/
├── run_tricast.py                 # 训练 / 评估 / smoke 入口
├── models/TriCast.py              # 模型整体组装
├── layers/TriCast.py              # MSDR、PatchEncoder、SpectralEncoder、AlignBlock、FiLM、FourierCrossAttn、CPRouter
├── data_provider/                 # ETT 与通用 CSV 数据加载
├── utils/
│   ├── llm_backbone.py            # 本地主干加载与身份校验
│   └── timefeatures.py            # 数据加载使用的 time-feature 编码
├── configs/
│   ├── llm_backbones.json         # 主干注册表（模型 id、本地路径、隐维、层数）
│   └── experiments/               # 28 份生成的实验配置
├── scripts/
│   ├── download_tricast_llm.py    # 显式下载主干的辅助脚本
│   ├── generate_tricast_experiments.py
│   ├── run_tricast_matrix.py      # 批量运行器（支持 dry-run）
│   ├── TriCast_stages.ps1         # 阶段 0–5 检查（Windows）
│   └── tricast/                   # 56 个生成的 PowerShell 训练/测试脚本
├── tests/                         # 模块测试与多主干测试
├── TriCast_IMPLEMENTATION.md      # 实现细节与分阶段设计
├── TriCast_MULTILLM.md            # 多主干与批量实验说明
├── requirements.txt
└── LICENSE
```

数据集、模型权重、检查点与运行日志**不**纳入版本管理。

### 环境依赖

- Python 3.11
- PyTorch、Transformers（Qwen2 需要 ≥ 4.37）、NumPy、pandas、scikit-learn、scipy

```bash
pip install -r requirements.txt
```

### 1. 准备数据集

将 ETT 的 CSV 放到 `dataset/ETT-small/`：

```text
dataset/ETT-small/ETTh1.csv
dataset/ETT-small/ETTh2.csv
dataset/ETT-small/ETTm1.csv
dataset/ETT-small/ETTm2.csv
```

ETT 为公开数据集，可从 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 获取。加载器沿用原始训练/验证/测试划分与训练集标准化。

### 2. 准备 LLM 主干

主干**只从本地路径加载**，不会自动联网下载。在 `configs/llm_backbones.json` 中为目标主干填写 `local_path`，或使用下载脚本：

```bash
python scripts/download_tricast_llm.py --model gpt2 --output ./llm_weights/gpt2 --register
```

注册表支持：`gpt2`、`gpt2-large`、`gpt2-xl`、`qwen2.5-0.5b`、`qwen2.5-1.5b`、`qwen2.5-3b`、`tinyllama-1.1b`。

### 3. 快速开始

以 GPT-2 + ETTh1 配置为例：

```bash
# 只检查主干与数据划分，不加载完整模型
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode inspect

# 前向/反向 smoke 检查
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode smoke --batch_size 1

# 训练（按验证集 MSE 保存最佳适配器）
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode train

# 评估已保存的检查点
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode eval \
  --checkpoint ./checkpoints/tricast_single/gpt2/ETTh1/best.pt \
  --output_dir ./logs/tricast_eval
```

输出（`best.pt`、`history.json`、`metrics.json`、`resolved_config.json`）写入 `output_dir`。

### 4. 分阶段消融

`--tricast_stage` 按顺序累积启用模块：

| 阶段 | 启用内容 |
| --- | --- |
| 0 | 时域分块编码 → 维度映射 → 冻结 LLM → 预测头 |
| 1 | + MSDR 突变分支（仅生成先验） |
| 2 | + 频谱编码与三个 AlignBlock |
| 3 | + FiLM 注入突变先验 |
| 4 | + 动态频谱门控与 Fourier 交叉注意力 |
| 5 | + CPRouter 与四个专家（完整模型） |

```bash
python run_tricast.py --config configs/experiments/gpt2_ETTh1.json --mode smoke --tricast_stage 3
```

Windows 下可一次跑完 0–5 检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/TriCast_stages.ps1
```

### 5. 批量实验

```bash
# 只预览命令，不执行
python scripts/run_tricast_matrix.py --mode train-test --dry_run

# 单个主干、指定预测长度
python scripts/run_tricast_matrix.py --mode train --models gpt2-xl --pred_lens 96 192

# 全部 7 种主干 × 4 个数据集 × 4 个预测长度，顺序执行
python scripts/run_tricast_matrix.py --mode train-test
```

每个模型/数据集也有独立的 PowerShell 脚本（Windows），位于 `scripts/tricast/`：

```powershell
./scripts/tricast/gpt2-xl_ETTh1_train.ps1 -DryRun
./scripts/tricast/gpt2-xl_ETTh1_train.ps1
```

如需重新生成配置与脚本：

```bash
python scripts/generate_tricast_experiments.py
```

> 该命令会覆盖全部 28 份配置和 56 个脚本，自定义后请勿随意重跑。

### 6. 测试

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

- `tests/test_tricast.py` — `MSDR`、`FiLM`、`FourierCrossAttn`、`CPRouter`、`PatchEncoder` 的模块行为。
- `tests/test_multillm.py` — 本地微型 GPT-2 / Qwen2 / Llama 夹具、精度处理、截断层数与身份校验。

### 说明

- 检查点只保存可训练适配器与解析后的配置；冻结主干权重从注册表重新加载。
- `metrics.json` 的 MSE/MAE 基于原始加载器训练集标准化后的尺度，并包含输出反归一化。
- 小规模调试跑出的数值仅用于验证流程，不代表论文结果。

### 致谢

TriCast 的实现受到 [Time-LLM](https://github.com/KimMeen/Time-LLM) 与 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 的启发并复用其数据加载接口。原始基线代码不在本仓库中。

### 许可证

[Apache License 2.0](LICENSE)
