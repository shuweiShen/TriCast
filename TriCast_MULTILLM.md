# 配置 LLM 与运行四个 ETT 数据集

模型主干现通过 Hugging Face AutoConfig/AutoModel 加载，时域映射、专家和预测头自动适配主干隐藏维度。GPT-2、Qwen2 和 Llama 架构共用相同 TriCast 模块。LLM 保持冻结，但保留输入适配器所需的反向传播计算图。

## 1 配置权重路径

编辑 `configs/llm_backbones.json`，为需要使用的模型填写 `local_path`。例如 TinyLlama 的条目：

```json
"tinyllama-1.1b": {
  "model_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
  "local_path": "D:/LLM/TinyLlama-1.1B",
  "model_type": "llama",
  "hidden_size": 2048,
  "full_layers": 22
}
```

路径可以是包含 `config.json` 和权重的模型目录，也可以是 Hugging Face 的 `models--...` 缓存目录。JSON 内建议使用正斜杠。相对路径以项目根目录为基准。仅填写目录，不要指向单个权重文件。

| 配置名称 | 默认模型仓库 | 隐维 | 完整层数 |
| --- | --- | --- | --- |
| gpt2-xl | openai-community/gpt2-xl | 1600 | 48 |
| gpt2-large | openai-community/gpt2-large | 1280 | 36 |
| gpt2 | openai-community/gpt2 | 768 | 12 |
| qwen2.5-0.5b | Qwen/Qwen2.5-0.5B | 896 | 24 |
| qwen2.5-1.5b | Qwen/Qwen2.5-1.5B | 1536 | 28 |
| qwen2.5-3b | Qwen/Qwen2.5-3B | 2048 | 36 |
| tinyllama-1.1b | TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | 22 |

TinyLlama 默认选择基础预训练版本。若使用同结构的 TinyLlama Chat 权重，请同步修改 model_id 和 local_path，训练与测试应使用完全相同的权重。

`local_path=null` 时依次尝试项目内 `llm_weights/<配置名称>`、项目内对应 HF 缓存目录以及用户已有 HF 缓存。加载器不会自动联网下载，也不会回退到另一种模型。启动时会验证架构、隐维及原始层数，打印真实加载位置、完整层数和使用层数。

若需要下载，可明确执行下载脚本，并自行选择存储路径：

```powershell
conda activate TimeSeries
python scripts/download_tricast_llm.py --model tinyllama-1.1b --output D:/LLM/TinyLlama-1.1B --register
```

`--register` 在成功下载后把路径写入注册表；其他模型同样可用。此脚本仅在你主动调用时下载。

## 2 实验配置文件

`configs/experiments/` 包含 7 种主干 × 4 个数据集，共 28 份 JSON，例如：

```text
gpt2-xl_ETTh1.json
gpt2-large_ETTh2.json
gpt2_ETTm1.json
qwen2.5-0.5b_ETTm2.json
tinyllama-1.1b_ETTh1.json
```

文件中的 `llm_name` 选择主干；`llm_layers=null` 使用全部预训练层。若显式指定 4，则仅使用前 4 层，属于截断模型实验。所有命令行参数优先于实验 JSON。

实验默认使用 672 步历史、96 步非重叠分块、完整 TriCast 模块、10 epochs、patience=3、AdamW、学习率 1e-4。单配置默认预测 96 步；批量脚本默认分别运行 96、192、336、720 步。

GPT-2 默认 float32；其余主干权重默认 bfloat16，适配器仍为 float32。若设备不支持 bf16，可在配置中将 `llm_dtype` 改为 float32。默认 batch size 分别为 4 和 2，主要用于本机显存；这与论文 batch size 16 不同，可自行调整。权重 dtype 与 `amp` 分开配置：前者控制冻结主干权重精度，后者控制 CUDA 混合精度上下文。

现有 TimeSeries 环境是 Transformers 4.46.3，已支持这些架构。requirements.txt 已同步为该版本；Qwen 官方说明旧于 4.37.0 的版本无法识别 qwen2，参见 [Qwen2.5 模型说明](https://huggingface.co/Qwen/Qwen2.5-0.5B)。不需要重新安装当前已可用的环境。

## 3 数据准备

确保目录中具有：

```text
dataset/ETT-small/ETTh1.csv
dataset/ETT-small/ETTh2.csv
dataset/ETT-small/ETTm1.csv
dataset/ETT-small/ETTm2.csv
```

ETTh 配置使用小时频率 h，ETTm 使用分钟频率 t。数据划分和训练集标准化沿用原仓库 ETT 加载器；新增窗口读取保留全部 7 个变量。训练集打乱，验证/测试不打乱、不丢尾批。

## 4 先检查单个模型

```powershell
python run_tricast.py --config configs/experiments/gpt2-xl_ETTh1.json --mode inspect
python run_tricast.py --config configs/experiments/gpt2-xl_ETTh1.json --mode smoke --batch_size 1
```

inspect 检查本地权重配置、权重文件是否存在和三个数据划分，不加载完整模型。smoke 检查真实数据前向、反向、参数更新及冻结主干。

## 5 每个模型与数据集的独立脚本

`scripts/tricast/` 包含 56 个 PowerShell 脚本，每个模型/数据集各有 train 和 test。例如 GPT2-xl 在 ETTh1：

```powershell
# 先预览命令，不训练。
./scripts/tricast/gpt2-xl_ETTh1_train.ps1 -DryRun

# 训练四个预测长度。训练结束会自动恢复最佳检查点并测试。
./scripts/tricast/gpt2-xl_ETTh1_train.ps1

# 单独重新测试四个预测长度，直接读取训练时保存的 best.pt。
./scripts/tricast/gpt2-xl_ETTh1_test.ps1
```

TinyLlama 在 ETTm2 的对应脚本为：

```powershell
./scripts/tricast/tinyllama-1.1b_ETTm2_train.ps1
./scripts/tricast/tinyllama-1.1b_ETTm2_test.ps1
```

其他模型、数据集使用相同命名规则。默认 Python 为 `D:\Anaconda\envs\TimeSeries\python.exe`，可通过 `-Python` 覆盖。

逐步调试可以限制为一个预测长度、前 4 层、少量批次，使用独立标签避免覆盖正式结果：

```powershell
./scripts/tricast/gpt2_ETTh1_train.ps1 -PredLens 96 -Layers 4 -BatchSize 2 -Epochs 1 -MaxBatches 2 -RunTag debug
./scripts/tricast/gpt2_ETTh1_test.ps1 -PredLens 96 -Layers 4 -BatchSize 2 -MaxBatches 2 -RunTag debug
```

MaxBatches 同时限制训练、验证及测试，得到的指标只用于调试。测试时 Layers 和 RunTag 必须与训练一致；BatchSize 可以改变。训练/测试脚本中的 MaxBatches=0 表示完整划分。

## 6 批量脚本

```powershell
# 预览全部 7×4×4 组实验的训练和独立测试命令。
python scripts/run_tricast_matrix.py --mode train-test --dry_run

# GPT2-xl 在四个数据集上分别训练 96 和 192 步预测。
python scripts/run_tricast_matrix.py --mode train --models gpt2-xl --pred_lens 96 192

# Qwen 0.5B 在 ETTh1、ETTh2 上训练后独立测试。
python scripts/run_tricast_matrix.py --mode train-test --models qwen2.5-0.5b --datasets ETTh1 ETTh2

# 全部模型、四个数据集、四个预测长度，顺序运行；遇错立即停止。
python scripts/run_tricast_matrix.py --mode train-test

# 单独测试全部已训练模型。
python scripts/run_tricast_matrix.py --mode test
```

在项目根目录运行以上命令。数据或权重缺失会报错，不会伪造结果。每组输出路径形如：

```text
checkpoints/tricast_matrix/gpt2-xl/ETTh1/sl672_pl96_layersfull/default/
    best.pt
    resolved_config.json
    history.json
    metrics.json
```

检查点记录主干名称、实际路径、配置校验值、权重文件大小与修改时间清单、实际层数及精度。测试时严格核对，防止相同隐藏维度的不同模型误加载同一适配器。文件移动或重新拷贝导致路径/修改时间变化也会触发检查；这项清单不是权重内容的完整密码学校验。默认 legacy GPT-2 检查点允许加载并提示身份未记录，其他模型不能使用旧的无身份检查点。

更改模型参数、数据设置、精度或随机种子后，请使用新的 RunTag；不同设置不应覆盖同一实验目录。再次运行配置生成器会重置生成的 28 份配置和 56 个脚本，不要在修改个人配置后无意重跑它。

## 7 验证范围

新增测试用本地随机小模型分别验证 GPT2Model、Qwen2Model、LlamaModel 的加载、float32/bfloat16 前向、梯度回传、全层/截断层数和错误模型身份拦截。它们验证架构兼容，不代表已经验证全部大型预训练权重。完整实验仍需准备各模型权重以及全部 ETT 数据。

本次已验证：6 项测试通过（覆盖三个主干家族）；56 个 PowerShell 脚本全部通过语法检查；完整批量计划生成 224 条命令；本地预训练 GPT-2 在 ETTh1 的实际训练和独立测试脚本跑通，重新加载后的测试指标一致。该检查采用前 4 层、每个划分前 2 批，仅用于验证运行链路。

核对时本项目仅发现 GPT-2 small 权重和 ETTh1.csv。其他 6 种预训练权重与 ETTh2/ETTm1/ETTm2 数据尚未准备，因此未宣称这些完整实验已运行。
