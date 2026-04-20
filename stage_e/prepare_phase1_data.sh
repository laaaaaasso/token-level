#!/bin/bash
# ===========================================================================
# Phase 1 数据准备 (debug集群用)
#
# 将 Step 2 的 CSV manifest 转为 CosyVoice2 训练格式:
#   manifest_ref.csv → parquet shards → train.data.list
#   manifest_cv.csv  → parquet shards → cv.data.list
# 同时生成训练配置 phase1_cosyvoice2.yaml
#
# 使用:
#   cd /data/zhenghao/repos/token-level && bash stage_e/prepare_phase1_data.sh
#
# 完成后在训练集群运行:
#   bash stage_e/launch_phase1_30h.sh
# ===========================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COSYVOICE_ROOT="/data/zhenghao/repos/CosyVoice"
CONDA_PYTHON="/data/zhenghao/miniforge3/envs/cosyvoice2/bin/python"

MULTI_DATA_DIR="${PROJECT_ROOT}/multi_dataset_30h"
AUDIO_DIR="${MULTI_DATA_DIR}/audios"
SPLITS_DIR="${MULTI_DATA_DIR}/splits"
OUTPUT_DIR="${PROJECT_ROOT}/phase1_outputs_30h"
DATA_DIR="${OUTPUT_DIR}/data"

# 训练超参 (写入config)
LEARNING_RATE="1e-5"
MAX_EPOCH=3
ACCUM_GRAD=2
GRAD_CLIP=5
SCHEDULER="constantlr"
WARMUP_STEPS=100
LOG_INTERVAL=20
SAVE_PER_STEP=-1

BASE_CONFIG="${COSYVOICE_ROOT}/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml"
MANIFEST_REF="${SPLITS_DIR}/manifest_ref.csv"
MANIFEST_CV="${SPLITS_DIR}/manifest_cv.csv"

echo "Phase 1 Data Preparation"
echo ""

# 检查输入
for f in "${MANIFEST_REF}" "${MANIFEST_CV}" "${BASE_CONFIG}"; do
    [ ! -f "$f" ] && echo "ERROR: $f not found" && exit 1
done
[ ! -d "${AUDIO_DIR}" ] && echo "ERROR: ${AUDIO_DIR} not found" && exit 1

mkdir -p "${DATA_DIR}"

# ── 1. CSV → Parquet ─────────────────────────────────────────────────

echo "Converting CSV manifests to parquet..."

export AUDIO_DIR DATA_DIR MANIFEST_REF MANIFEST_CV
export UTTS_PER_SHARD=500

${CONDA_PYTHON} << 'PYEOF'
import csv, json, os, sys, logging
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prep")

AUDIO_DIR = Path(os.environ["AUDIO_DIR"])
DATA_DIR = Path(os.environ["DATA_DIR"])
SHARD_SIZE = int(os.environ.get("UTTS_PER_SHARD", "500"))

def csv_to_parquet(csv_path, out_dir, prefix):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f): rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards, valid, skip = [], 0, 0
    for si in range(0, len(rows), SHARD_SIZE):
        recs = []
        for row in rows[si:si+SHARD_SIZE]:
            fn = row["audio_path"].replace("\\","/").split("/")[-1]
            ap = AUDIO_DIR / fn
            if not ap.exists(): skip += 1; continue
            text = str(row.get("text","")).strip()
            if not text: skip += 1; continue
            with open(ap, "rb") as f: ab = f.read()
            recs.append({"utt": ap.stem, "audio_data": ab, "text": text})
            valid += 1
        if recs:
            sp = out_dir / f"{prefix}_{si//SHARD_SIZE:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(recs), sp)
            shards.append(sp)
    logger.info("%s: %d valid, %d skipped, %d shards", prefix, valid, skip, len(shards))
    return shards

def write_list(shards, path):
    with open(path, "w") as f:
        for s in shards: f.write(str(s.resolve()) + "\n")

train_shards = csv_to_parquet(os.environ["MANIFEST_REF"], DATA_DIR/"train", "train")
cv_shards = csv_to_parquet(os.environ["MANIFEST_CV"], DATA_DIR/"cv", "cv")
if not train_shards or not cv_shards:
    logger.error("Failed!"); sys.exit(1)
write_list(train_shards, DATA_DIR/"train.data.list")
write_list(cv_shards, DATA_DIR/"cv.data.list")
json.dump({"train_shards":len(train_shards),"cv_shards":len(cv_shards)},
          open(DATA_DIR/"prepare_summary.json","w"), indent=2)
logger.info("Done.")
PYEOF

echo "  train.data.list: $(wc -l < "${DATA_DIR}/train.data.list") shards"
echo "  cv.data.list:    $(wc -l < "${DATA_DIR}/cv.data.list") shards"

# ── 2. 生成训练配置 ──────────────────────────────────────────────────

echo ""
echo "Generating training config..."

PHASE1_CONFIG="${OUTPUT_DIR}/phase1_cosyvoice2.yaml"

${CONDA_PYTHON} << CONFEOF
import re
with open("${BASE_CONFIG}", "r") as f:
    lines = f.readlines()
in_tc, in_oc, in_sc = False, False, False
for i, line in enumerate(lines):
    s = line.strip()
    if line and not line[0].isspace() and ":" in line:
        if s.startswith("train_conf:"): in_tc, in_oc, in_sc = True, False, False; continue
        elif in_tc: in_tc, in_oc, in_sc = False, False, False
    if not in_tc: continue
    if re.match(r"^\s{4}optim_conf:\s*$", line): in_oc, in_sc = True, False; continue
    if re.match(r"^\s{4}scheduler_conf:\s*$", line): in_sc, in_oc = True, False; continue
    if re.match(r"^\s{4}[a-zA-Z_]\w*:", line) and not re.match(r"^\s{4}(optim_conf|scheduler_conf):", line): in_oc = in_sc = False
    if in_oc and re.match(r"^\s{8}lr:", line): lines[i] = "        lr: ${LEARNING_RATE}\n"
    if in_sc and re.match(r"^\s{8}warmup_steps:", line): lines[i] = "        warmup_steps: ${WARMUP_STEPS}\n"
    if re.match(r"^\s{4}max_epoch:", line): lines[i] = "    max_epoch: ${MAX_EPOCH}\n"
    if re.match(r"^\s{4}log_interval:", line): lines[i] = "    log_interval: ${LOG_INTERVAL}\n"
    if re.match(r"^\s{4}save_per_step:", line): lines[i] = "    save_per_step: ${SAVE_PER_STEP}\n"
    if re.match(r"^\s{4}accum_grad:", line): lines[i] = "    accum_grad: ${ACCUM_GRAD}\n"
    if re.match(r"^\s{4}grad_clip:", line): lines[i] = "    grad_clip: ${GRAD_CLIP}\n"
    if re.match(r"^\s{4}scheduler:", line): lines[i] = "    scheduler: ${SCHEDULER}\n"
with open("${PHASE1_CONFIG}", "w") as f: f.writelines(lines)
CONFEOF

echo "  Config: ${PHASE1_CONFIG}"
echo "  lr=${LEARNING_RATE}, epoch=${MAX_EPOCH}, accum=${ACCUM_GRAD}, warmup=${WARMUP_STEPS}"
echo ""
echo "Data preparation complete. Now run on training cluster:"
echo "  cd ${PROJECT_ROOT} && bash stage_e/launch_phase1_30h.sh"
