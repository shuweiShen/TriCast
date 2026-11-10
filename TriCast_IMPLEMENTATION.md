# TriCast 实现与分阶段运行

多主干配置及四个 ETT 数据集的训练/测试脚本请优先阅读 [TriCast_MULTILLM.md](TriCast_MULTILLM.md)。以下保留早期 GPT-2 调试记录；当前不指定 llm_layers 时默认使用全部层。

TriCast 的入口是 `run_tricast.py`，模型实现位于 `models/TriCast.py` 与 `layers/TriCast.py`。本机环境为 `D:\Anaconda\envs\TimeSeries\python.exe`。LLM 只从 `configs/llm_backbones.json` 配置的本地路径加载，不联网。

## 分阶段检查

在项目目录的 PowerShell 中运行：

```powershell
conda activate TimeSeries
python -m unittest discover -s tests -p test_tricast.py -v
powershell -ExecutionPolicy Bypass -File scripts/TriCast_stages.ps1
```

脚本依次检查以下阶段，遇到失败立即停止。每个阶段检查真实 ETTh1 数据的输出形状、有限值、反向传播、适配器更新、GPT-2 冻结与确定性推理；结果保存在 `logs/tricast_stages/smoke_stage*.json`。

| 阶段 | 累计启用内容 |
| --- | --- |
| 0 | 非重叠时域编码 → 维度映射 → 冻结 GPT-2 → 预测头 |
| 1 | MSDR 与突变分支，仅验证先验生成；尚未注入预测 |
| 2 | 频域编码、三个独立 AlignBlock；先验融合前的编码检查 |
| 3 | FiLM 将突变先验注入时域 token |
| 4 | 动态频谱门控与 FourierCrossAttn |
| 5 | CPRouter 和四个专家，完整模型 |

阶段 0 是本文架构的逐步构建起点，并不等价于原始 Time-LLM 基线。阶段 1/2 中部分先验分支尚未进入预测计算，训练时没有预测梯度属于预期行为。逐阶段检查是错误定位工具，不能替代各模块独立移除的消融实验。

单独定位某阶段：

```powershell
python run_tricast.py --mode smoke --tricast_stage 3 --batch_size 2
```

## 小规模训练与完整配置

先用少量批次检查训练、验证、保存、重新载入和测试的整条流程：

```powershell
python run_tricast.py --mode train --tricast_stage 5 --train_epochs 1 --max_batches 5 --output_dir checkpoints/TriCast_debug
```

上面的验证和测试也只检查前 5 个批次，指标仅供调试。完整训练应移除 `--max_batches`。

论文主要参数配置（显存不足时降低 batch size）：

```powershell
python run_tricast.py --mode train --tricast_stage 5 --seq_len 672 --pred_len 96 --patch_len 96 --llm_layers 12 --n_heads 8 --num_experts 4 --msdr_scales 16 32 64 --batch_size 16 --train_epochs 10 --patience 3 --learning_rate 0.0001 --output_dir checkpoints/TriCast_ETTh1_672_96
```

预测步长分别可改为 96、192、336、720，每个实验请使用独立输出目录。默认不启用混合精度；支持 CUDA `--amp` 使用 bf16。默认调试模型是 4 层 GPT-2、历史长度 192、batch size 4。它不对应论文完整实验。

每次训练按验证 MSE 保存最佳适配器至 `best.pt`，保存 `history.json`，恢复最佳适配器后才计算测试 MSE/MAE 并写入 `metrics.json`。GPT-2 权重从本地注册表恢复，检查点不重复保存冻结权重，也不删除已有检查点。分数在原数据加载器训练集标准化后的尺度上计算，包含输出实例反归一化。验证、测试不打乱、不丢弃最后一个批次。

独立评估必须传入训练时相同的模型参数，例如上述完整实验：

```powershell
python run_tricast.py --mode eval --checkpoint checkpoints/TriCast_ETTh1_672_96/best.pt --seq_len 672 --pred_len 96 --patch_len 96 --llm_layers 12 --batch_size 4 --output_dir logs/TriCast_eval
```

## 论文公式与实现选择

- `layers/TriCast.py` 的 MSDR 对照式 (1)–(12)：左右窗口排除当前时间点；Median、1.4826×MAD；水平、对数波动比、自相关差异；三维均方根融合；均值×exp(−跨尺度标准差/均值)；最后使用 Q75/Q25 的指数归一化。所有统计均只取当前输入窗口，不读取未来预测标签。
- `AlignBlock`、FiLM、FourierCrossAttn、CPRouter 分别对应式 (13)–(17)。FiLM 使用 `1+tanh(gamma)` 并进行 LayerNorm。CPRouter 权重只由突变与频谱先验决定；所有专家输出经过 softmax 加权求和，没有引入论文未描述的 top-k 稀疏路由或额外损失。
- 时域和突变模态使用独立线性非重叠分块投影，默认 P=96。不能整除的尾部复制末值补齐。频域使用整个输入的正交归一化 rFFT，以实部/虚部作为二维频点特征，再线性投影；保留相位。
- 论文未写清的实现细节采用以下约定：MSDR 在每个样本、每个变量的输入窗口内计算全局 MAD 噪声及分位数；边界复制填充；最大自相关滞后默认为 min(8, h/2)，相关分母下界 rho0=0.1；跨尺度标准差使用总体标准差；数值分母加 epsilon。这些是可核对的工程选择，不能声称与作者未提供的实现完全相同。
- 频谱门控为时域 token 均值的线性映射经 sigmoid 得到逐通道门控；频点作为 K/V。路由前分别池化突变与频谱 token，拼接后 MLP 融合为状态条件。专家使用瓶颈 MLP；GPT-2 后使用线性降维及展平预测头。论文未明确这些细节，属于当前复现约定。隐维 D=128、预测头维度 128、dropout=0.1、AdamW weight decay=0.01 也作为可配置默认值。
- 新架构按论文采用融合 token → 冻结 GPT-2，没有保留原 Time-LLM 的文本 prompt 与词表重编程层。GPT-2 始终 eval，但计算图保留，确保前端适配器能通过冻结主干获得梯度。
- 原数据加载器按变量展开样本；新入口复用其训练集 scaler 和时间划分，并按窗口取所有变量。参数在变量间共享，每个变量作为独立 GPT-2 序列处理。`features=MS` 时仅目标变量进入损失和指标。
- 当前沿用仓库已有数据类型。Solar-Energy 没有专用加载器，未将通用 CSV 划分冒充论文 Solar 实验协议。

这些检查用于确认实现与训练链路可工作，不代表已复现论文表格中的精度或性能收益。完整七数据集实验尚需逐项运行。

## 本机验证记录 2026-11-09

- 使用本机 conda `TimeSeries` 与 CUDA，4 项模块行为测试通过；阶段 0–5 的真实数据 smoke 全部通过。
- 完整模型在 672 步历史、12 层 GPT-2、batch size 2、bf16 下通过前向、反向及参数更新检查，输出形状为 `[2, 96, 7]`。
- 轻量训练使用 ETTh1、192→96、4 层 GPT-2、10% 训练数据、batch size 8、1 epoch。验证与测试使用各自完整划分，测试 MSE=0.725045、MAE=0.556868。结果保存在 `checkpoints/TriCast_ETTh1_quick/`，最佳适配器保存在同目录 `best.pt`（该实验产物未随仓库发布）。
- 独立评估进程重载调试检查点后，测试 MSE/MAE 与保存时完全一致。
