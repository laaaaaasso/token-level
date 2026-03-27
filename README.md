# Token-Level Selective Training for CosyVoice2

基于 **RHO-1** 思路对 CosyVoice2 text-speech LM 做 token-level selective training 的完整实现。

核心思路：不是所有 speech token 都同等重要。通过比较 target model 和 frozen reference model 的 per-token loss 差异（excess loss），筛选出最有训练价值的 token，只对这些 token 做 backprop。

## 整体流程

```
Phase 0          Phase 1            Phase 2              Phase 3
数据准备  ──→  训练 Reference  ──→  半离线 Token  ──→  Selective
D_ref/D_train     Model (RM)         Scoring            Training
```

- **Phase 0**：从原始音频数据中分层筛选，划分 D_ref（高质量子集）和 D_train（全量训练集）
- **Phase 1**：在 D_ref 上训练 Reference Model，作为”基线水平”的参照
- **Phase 2**：对 D_train 做双模型前向（target + frozen RM），逐 speech token 计算 excess loss，生成 top-k% mask
- **Phase 3**：训练时只对被 mask 选中的 speech token 计算 loss 并回传梯度

## 项目结构

```text
.
├── phase0/                     # Phase 0: 数据准备与分层
│   ├── pipeline.py
│   └── tokenizers.py
├── phase1/                     # Phase 1: Reference Model 训练
│   ├── prepare_data.py         #   D_ref → CosyVoice2 parquet 格式
│   └── export_checkpoint.py    #   导出冻结的 RM checkpoint
├── phase2/                     # Phase 2: 半离线 token scoring
│   ├── scoring.py              #   核心计算：per-token loss, excess loss, top-k mask
│   └── score_tokens.py         #   主脚本：双模型前向 + 打分 + 落盘
├── phase3/                     # Phase 3: selective training
│   ├── mask_loader.py          #   加载 Phase 2 产出的 token mask
│   ├── selective_loss.py       #   selective SLM loss 计算
│   └── train.py                #   主训练脚本（复用 CosyVoice2 训练基建）
├── scripts/                    # 启动脚本
│   ├── download_model.sh       #   下载 CosyVoice2-0.5B 预训练模型
│   ├── phase1_train.sh         #   Phase 1 全流程启动
│   ├── phase2_score.sh         #   Phase 2 打分启动
│   └── phase3_train.sh         #   Phase 3 selective training 启动
├── prompts and Report/         # 各阶段的设计文档
├── main.py                     # Phase 0 音频显著性打标入口
├── phase0_main.py              # Phase 0 数据划分入口
└── phase1_main.py              # Phase 1 辅助入口
```

## 环境要求

- Python 3.10+
- CUDA GPU（训练和打分阶段）
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 仓库（克隆到本机）
- conda 环境 `cosyvoice2`（含 PyTorch, transformers, onnxruntime 等）

```bash
# 下载预训练模型
bash scripts/download_model.sh
```

## 快速开始

### Phase 0：数据准备

```bash
# 音频特征提取 + 显著性打标
python main.py --metadata crema_data_1000/metadata.csv --outdir crema_outputs

# D_ref / D_train 划分
python phase0_main.py \
  --train-manifest crema_data_1000/metadata.csv \
  --ref-manifest crema_outputs/high_score_audios.csv \
  --outdir phase0_outputs
```

### Phase 1：训练 Reference Model

```bash
# Dry run（只准备数据和配置）
bash scripts/phase1_train.sh

# 实际训练
bash scripts/phase1_train.sh --run
```

输出：`phase1_outputs/rm/rm_frozen.pt`

### Phase 2：Token Scoring

```bash
# Dry run
bash scripts/phase2_score.sh

# 实际打分
bash scripts/phase2_score.sh
```

输出：`phase2_outputs/token_scores.pt`（每个样本的 speech token mask + excess loss score）

### Phase 3：Selective Training

```bash
# Dry run（准备配置，打印命令）
bash scripts/phase3_train.sh

# 实际训练
bash scripts/phase3_train.sh --run

# 自定义参数
bash scripts/phase3_train.sh --run --epoch 5 --lr 1e-5
```

## 关键设计

### Excess Loss

对每个 speech token μ_i：

```
L_delta(μ_i) = L_target(μ_i) - L_ref(μ_i)
```

L_delta 越大 → target 相对 RM 损失越高 → 该 token 更值得训练。

### Token Mask

选择 L_delta 最高的 top-k% token（默认 k=60%）：

```
I_k%(μ_i) = 1,  if μ_i ∈ top-k% by L_delta
             0,  otherwise
```

### Selective SLM Loss

```
L_SLM = (1 / N_selected) * Σ I_k%(μ_i) * CE(μ_i)
```

只有被选中的 speech token 贡献 loss 和梯度。Text token、padding、未选中的 speech token 均不参与。

### 对齐保证

- Phase 2 和 Phase 3 均强制 **unistream** 序列布局（text 在前，speech 在后）
- Phase 2 双模型共享同一套 speech token（提取一次，避免 ONNX 浮点差异）
- Phase 3 按 `utt` id 查找对应 mask，speech token 位置严格对齐

## 配置说明

各阶段的 launch script 支持的主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--run` | false | 是否实际执行（默认 dry run） |
| `--gpus` | 1 | GPU 数量 |
| `--epoch` | 3 | 最大训练轮数 |
| `--lr` | Phase 依赖 | 学习率（Phase 1: 1e-5, Phase 3: 5e-6） |
| `--topk` | 0.6 | Phase 2 top-k% 比例 |
| `--no-amp` | - | 禁用混合精度训练 |

## Phase 0 补充说明

Phase 0 基于音频特征的统计检验做显著性打标：

1. 数据集层面做组间检验（t-test / Mann-Whitney U）
2. 单条样本判断是否落入其标签组的典型区间
3. 映射为 `gender_sig / age_sig / emotion_sig` 标记
4. `sig_score >= 2` 的样本进入 D_ref

```bash
# 显著性打标
python main.py --metadata metadata.csv --outdir outputs

# 可选参数
#   --alpha 0.05          显著性阈值
#   --interval-method std 区间方法 (std / iqr)
#   --std-multiplier 1.5  std 区间倍数
```

## 依赖的外部项目

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)：阿里巴巴的 TTS 系统，提供 text-speech LM 训练基建
- [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)：预训练模型权重
