# CosyVoice2 + RHO-1 Token-Level Selective Training — 训练全景解读

> 本文档详细记录了各阶段的训练数据、训练产物和启动脚本方法。

---

## 一、训练使用的数据

### 原始数据源
- **CREMA-D 数据集**（1000 条子集）：位于 `/data/zhenghao/repos/token-level/crema_data_1000/`
  - `audios/` 下有 **1000 个 .wav 文件**，是情感语音录音（如 `1046_IEO_SAD_MD.wav`）
  - `metadata.csv` 是原始清单
  - 文本内容很短，都是 3 字母缩写句子（如 "IEO"、"ITS"、"ITH"），平均 3 个 text token

### Phase 0 分层后的数据
经过 Phase 0 处理后，1000 条样本被分为：
- **`D_train`**（正式训练集）：**1000 条**（全部样本，`phase0_manifest_train.jsonl`）
- **`D_ref`**（高质量参考子集）：**849 条**（`phase0_manifest_ref.jsonl`，占 85%）
- 每条样本 tokenize 后平均 **3 个 text token (y)** + **125 个 speech token (μ)**
- Speech tokenizer：CosyVoice2 SpeechTokenizer，vocab_size=1024，sample_rate=16kHz
- **0 条失败样本**

### 各阶段实际使用的数据格式
Phase 1/2/3 不直接读 JSONL manifest，而是先转成 **Parquet 格式**（CosyVoice2 训练框架要求），通过 `train.data.list` 索引文件指向 parquet 分片：

| 阶段 | 训练数据 | 验证数据 |
|------|---------|---------|
| Phase 1 | `D_ref` 的 807 条（2 个 parquet shard） | `D_ref` 的 42 条（1 个 parquet shard），5% val split |
| Phase 2 | `D_train` 全部 1000 条（2 个 parquet shard） | 无（只做推理打分） |
| Phase 3 | `D_train` 全部 1000 条（复用 phase2 的 parquet） | 复用 Phase 1 的 cv 数据 |

---

## 二、训练产物

### Phase 0 产物（`phase0_outputs/`）
| 文件 | 说明 |
|------|------|
| `phase0_manifest_train.jsonl` (761K) | D_train 全部 1000 条的 (y, μ) manifest |
| `phase0_manifest_ref.jsonl` (649K) | D_ref 849 条的 (y, μ) manifest |
| `phase0_manifest_all.jsonl` (761K) | 全部样本合集 |
| `phase0_failures.jsonl` (0B) | 失败样本记录（空，全部成功） |
| `phase0_summary.json` | 统计摘要 |
| `phase0/` (symlink) | token 数据，链接到 `/data/zhenghao/exps/token-level-outputs/phase0` |

### Phase 1 产物（`phase1_outputs/`）— Reference Model
| 文件 | 说明 |
|------|------|
| `checkpoints/init.pt` (1.9G) | 初始 checkpoint（加载预训练权重后保存） |
| `checkpoints/epoch_0_whole.pt` (1.9G) | 第 0 epoch 结束时的 checkpoint |
| `checkpoints/epoch_1_whole.pt` (1.9G) | 第 1 epoch |
| `checkpoints/epoch_2_whole.pt` (1.9G) | 第 2 epoch（最终） |
| **`rm/rm_frozen.pt`** (1.9G) | **冻结的 Reference Model**，Phase 2/3 核心依赖 |
| `rm/rm_best.pt` (1.9G) | 验证 loss 最优的 RM |
| `rm/rm_last.pt` (1.9G) | 最后一个 epoch 的 RM |
| `phase1_cosyvoice2.yaml` | 训练配置（完整 CosyVoice2 配置） |
| `tensorboard/` | TensorBoard 日志 |
| `data/` | 转换后的 parquet 训练数据 |

### Phase 2 产物（`phase2_outputs/`）— Token Scoring
| 文件 | 说明 |
|------|------|
| **`token_scores.pt`** (1.8M) | 核心产物：每条样本的 per-token excess loss + top-k mask |
| `phase2_summary.json` | 打分统计摘要 |
| `data/` | D_train 的 parquet 数据（供 Phase 3 复用） |

`token_scores.pt` 内部结构：
```python
{
  'results': [
    {
      'utt': 'sample_id',
      'speech_token_score': tensor(...),  # L_delta 每个 speech token 的 excess loss
      'speech_token_mask': tensor(...),   # bool, top-60% 为 True
      'target_loss': tensor(...),         # target model per-token loss
      'ref_loss': tensor(...),            # ref model per-token loss
    },
    ...  # 共 1000 条
  ],
  'topk_ratio': 0.6,
  'total_tokens': 63901,
  'selected_tokens': 38299,          # 实际选中 ~59.9%
  'actual_ratio': 0.5993,
}
```

### Phase 3 产物（`phase3_outputs/`）— Selective Training
| 文件 | 说明 |
|------|------|
| `checkpoints/init.pt` (1.9G) | 初始 checkpoint |
| `checkpoints/epoch_0_whole.pt` (1.9G) | 第 0 epoch selective training 结果 |
| `checkpoints/epoch_1_whole.pt` (1.9G) | **最终 target model checkpoint** |
| `phase3_cosyvoice2.yaml` | 训练配置 |
| `tensorboard/` | TensorBoard 日志 |

---

## 三、启动训练脚本的方法

### Phase 0：数据准备
```bash
cd /data/zhenghao/repos/token-level

python phase0_main.py \
  --train-manifest crema_data_1000/metadata.csv \
  --ref-manifest <ref_manifest_path> \
  --outdir phase0_outputs \
  --audio-base-dir crema_data_1000/audios \
  --speech-sample-rate 16000 \
  --speech-vocab-size 1024
```

### Phase 1：训练 Reference Model

**步骤 1 — 准备数据**（JSONL → Parquet）：
```bash
python phase1_main.py prepare \
  --ref-manifest phase0_outputs/phase0_manifest_ref.jsonl \
  --audio-dir /data/zhenghao/data/crema_data_1000/audios \
  --output-dir phase1_outputs/data
```

**步骤 2 — 训练 RM**（直接用生成好的脚本）：
```bash
cd /data/zhenghao/repos/CosyVoice
bash /data/zhenghao/repos/token-level/phase1_outputs/phase1_train_command.sh
```
核心命令本质是：
```bash
export PYTHONPATH="/data/zhenghao/repos/CosyVoice:/data/zhenghao/repos/CosyVoice/third_party/Matcha-TTS"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  /data/zhenghao/repos/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model llm \
  --config phase1_outputs/phase1_cosyvoice2.yaml \
  --train_data phase1_outputs/data/train.data.list \
  --cv_data phase1_outputs/data/cv.data.list \
  --qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/CosyVoice2-0.5B \
  --checkpoint pretrained_models/CosyVoice2-0.5B/llm.pt \
  --model_dir phase1_outputs/checkpoints \
  --tensorboard_dir phase1_outputs/tensorboard \
  --num_workers 2 --prefetch 100 --pin_memory --use_amp
```
训练配置：Adam lr=1e-5, warmup=100步, **3 epochs**, grad_clip=5, accum_grad=2

**步骤 3 — 导出冻结 RM**：
```bash
python phase1_main.py export \
  --model-dir phase1_outputs/checkpoints \
  --output-dir phase1_outputs/rm
```

### Phase 2：Token Scoring
```bash
cd /data/zhenghao/repos/token-level

python -m phase2.score_tokens \
  --train-data-list phase2_outputs/data/train.data.list \
  --target-ckpt phase1_outputs/rm/rm_frozen.pt \
  --ref-ckpt phase1_outputs/rm/rm_frozen.pt \
  --config phase1_outputs/phase1_cosyvoice2.yaml \
  --qwen-pretrain-path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --onnx-path pretrained_models/CosyVoice2-0.5B \
  --output-dir phase2_outputs \
  --topk-ratio 0.6
```
> **注意**：当前实验中 target-ckpt 和 ref-ckpt 用的是同一个 `rm_frozen.pt`（意味着首轮 scoring 时 target 还没有独立训练，L_delta 反映的是初始差异）。耗时约 **121 秒**。

### Phase 3：Selective Training
```bash
cd /data/zhenghao/repos/token-level
bash phase3_outputs/phase3_train_command.sh
```
核心命令：
```bash
export PYTHONPATH="/data/zhenghao/repos/CosyVoice:/data/zhenghao/repos/CosyVoice/third_party/Matcha-TTS:/data/zhenghao/repos/token-level"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m phase3.train \
  --train_engine torch_ddp \
  --config phase3_outputs/phase3_cosyvoice2.yaml \
  --train_data phase2_outputs/data/train.data.list \
  --cv_data phase1_outputs/data/cv.data.list \
  --qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --onnx_path pretrained_models/CosyVoice2-0.5B \
  --mask_path phase2_outputs/token_scores.pt \
  --checkpoint phase1_outputs/rm/rm_frozen.pt \
  --model_dir phase3_outputs/checkpoints \
  --tensorboard_dir phase3_outputs/tensorboard \
  --num_workers 2 --prefetch 100 --pin_memory --use_amp
```
训练配置：Adam lr=**5e-6**（比 Phase 1 更小）, warmup=50步, **2 epochs**, grad_clip=5, accum_grad=2

关键区别：Phase 3 使用自定义的 `phase3.train` 模块而非 CosyVoice 原生 `train.py`，因为它通过 **monkey-patch** `model.forward` 注入了 selective loss 逻辑——只对 Phase 2 选出的 top-60% speech token 计算并回传梯度。

---

## 四、关键预训练模型

位于 `pretrained_models/CosyVoice2-0.5B/`：

| 文件 | 用途 |
|------|------|
| `llm.pt` | CosyVoice2 text-speech LM 预训练权重（Phase 1 初始化起点） |
| `CosyVoice-BlankEN/` | Qwen2 tokenizer 配置 |
| `speech_tokenizer_v2.onnx` | Speech tokenizer（Phase 0 使用） |
| `campplus.onnx` | 说话人嵌入模型 |
| `flow.pt` / `hift.pt` | CFM 和 vocoder 权重（本项目不修改） |
