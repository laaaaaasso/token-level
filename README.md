# Audio Significance Pipeline

基于组间统计检验（`t-test` / `Mann-Whitney U`）的音频标签显著性打标脚本。

核心统计逻辑：
1. 先在数据集层面做组间检验（而非逐条样本检验）。
2. 再判断单条样本是否落入其标签组的典型区间（默认 `mean ± 1.5*std`，也支持 IQR）。
3. 最终映射为 `gender_sig / age_sig / emotion_sig` 的 `0/1` 标记。

## 项目结构

```text
.
├─ main.py
├─ feature_extraction.py
├─ stats_validation.py
├─ scoring.py
├─ utils.py
├─ requirements.txt
├─ README.md
└─ examples/
   └─ create_demo_dataset.py
```

## 环境要求

- Python 3.10+

安装依赖：

```bash
pip install -r requirements.txt
```

可选增强库（未安装时自动降级）：
- `praat-parselmouth`（HNR / jitter / shimmer）
- `pyworld`（可扩展更稳健 F0）

## 输入格式

`metadata.csv` 至少包含：
- `audio_path`
- `text`
- `gender_label`（例如 `male/female`）
- `age_label`（支持映射到 `young/old`）
- `emotion_label`（`neutral/happy/sad/angry/fear/surprised`）

## Phase 0（数据准备与分层）

主入口：

```bash
python phase0_main.py --train-manifest crema_data_1000/metadata.csv --ref-manifest crema_outputs_1000_tight_std15/high_score_audios.csv --outdir phase0_outputs
```

输出：

- `phase0_manifest_all.jsonl`
- `phase0_manifest_train.jsonl`
- `phase0_manifest_ref.jsonl`
- `phase0_failures.jsonl`
- `phase0_summary.json`

## Phase 1（Reference Model 训练）

主入口：

```bash
python phase1_main.py --ref-manifest phase0_outputs/phase0_manifest_ref.jsonl --output-dir phase1_outputs
```

说明：上面命令默认只做 `D_ref` 数据准备与训练命令生成（不启动训练）。

在有 GPU 的机器上启动 CosyVoice2 LLM 训练：

```bash
python phase1_main.py --ref-manifest phase0_outputs/phase0_manifest_ref.jsonl --output-dir phase1_outputs --qwen-pretrain-path /path/to/CosyVoice2-0.5B/CosyVoice-BlankEN --init-checkpoint /path/to/CosyVoice2-0.5B/llm.pt --num-gpus 1 --run-training
```

主要输出：

- `phase1_ref_train.data.list`
- `phase1_ref_cv.data.list`
- `phase1_cosyvoice2_ref.yaml`
- `phase1_train_command.txt`
- `phase1_train_summary.json`
- `checkpoints/*.pt`（训练后）
- `rm_last.pt`（训练成功后导出）
- `rm_best.pt`（训练成功后导出）
- `rm_frozen.pt`（训练成功后导出，供后续 Phase 2 读取）

## 运行方式

```bash
python main.py --metadata metadata.csv --outdir outputs
```

可选参数：
- `--sample-rate`：默认 `16000`
- `--alpha`：显著性阈值，默认 `0.05`
- `--interval-method`：区间方法，`std` 或 `iqr`，默认 `std`
- `--std-multiplier`：当 `interval-method=std` 时生效，默认 `1.5`
- `--iqr-multiplier`：当 `interval-method=iqr` 时生效，默认 `1.5`
- `--age-map-json`：自定义年龄映射 JSON
- `--verbose`：输出 debug 日志

## 最小可运行示例

1. 生成示例数据：

```bash
python examples/create_demo_dataset.py --outdir demo_data
```

2. 运行主流程：

```bash
python main.py --metadata demo_data/metadata.csv --outdir demo_outputs
```

## 使用 Hugging Face 的 CREMA-D（1000条随机样本）

1. 生成并下载 1000 条真实样本：

```bash
python examples/prepare_crema_d_hf_sample.py --outdir crema_data_1000 --n-samples 1000 --seed 42 --age-threshold 40
```

2. 跑统计验证：

```bash
python main.py --metadata crema_data_1000/metadata.csv --outdir crema_outputs_1000
```

如果想进一步收紧样本筛选，可使用更严格区间参数，例如：

```bash
python main.py --metadata crema_data_1000/metadata.csv --outdir crema_outputs_1000_tight --interval-method std --std-multiplier 1.5
```

## 输出文件

运行后会生成：

1. `features.csv`：每条音频的提取特征
2. `group_test_results.csv`：组间显著性检验结果
3. `audio_significance_results.csv`：原始数据 + 特征 + 三个显著性标志 + `sig_score`
4. `high_score_audios.csv`：`sig_score >= 2` 的高分样本
5. `summary.json`：总样本数、显著特征数量、分数分布等摘要
6. 图像：
   - `gender_f0_boxplot.png`
   - `age_feature_boxplot.png`
   - `emotion_feature_boxplot.png`
   - `sig_score_hist.png`

## 显著性规则

- `gender_sig`：
  - 数据集层面 `f0_mean` 组间检验显著，且样本 `f0_mean` 落在对应性别区间 -> `1`
- `age_sig`：
  - 在 `f0_mean` / `pause_ratio` 中，满足“检验显著 + 落入区间”的特征数 `>=1` -> `1`
- `emotion_sig`：
  - 对所属情绪的 one-vs-rest 显著特征中，样本满足区间约束的特征数 `>=2` -> `1`

总分：

```text
sig_score = gender_sig + age_sig + emotion_sig
```
