#!/bin/bash
# =============================================================================
# Phase 2: Semi-offline Token Scoring
#
# Runs dual-model forward (target + frozen reference) on D_train,
# computes per-speech-token excess loss, generates top-k% mask.
#
# Prerequisites:
#   - Phase 1 completed (RM checkpoint available)
#   - D_train parquet data prepared
#
# Usage:
#   bash scripts/phase2_score.sh
#   bash scripts/phase2_score.sh --topk 0.6 --target-ckpt path/to/target.pt
# =============================================================================

set -euo pipefail

# ─── Default Configuration ───────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COSYVOICE_ROOT="/data/zhenghao/repos/CosyVoice"
PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_models/CosyVoice2-0.5B"

# Data — use D_train (all training data including ref subset)
TRAIN_MANIFEST="${PROJECT_ROOT}/phase0_outputs/phase0_manifest_train.jsonl"
AUDIO_DIR="/data/zhenghao/data/crema_data_1000/audios"

# Model paths
CONFIG="${PROJECT_ROOT}/phase1_outputs/phase1_cosyvoice2.yaml"
QWEN_PRETRAIN="${PRETRAINED_MODEL_DIR}/CosyVoice-BlankEN"
ONNX_PATH="${PRETRAINED_MODEL_DIR}"

# For Phase 2 demonstration: use RM as both target and ref
# In real usage, target_ckpt would be the current target model checkpoint
TARGET_CKPT="${PROJECT_ROOT}/phase1_outputs/rm/rm_frozen.pt"
REF_CKPT="${PROJECT_ROOT}/phase1_outputs/rm/rm_frozen.pt"

# Output
OUTPUT_DIR="${PROJECT_ROOT}/phase2_outputs"
DATA_DIR="${OUTPUT_DIR}/data"

# Scoring params
TOPK_RATIO=0.6
DEVICE="cuda:0"
NUM_WORKERS=0
PREFETCH=100

# Conda
CONDA_PYTHON="/data/zhenghao/miniforge3/envs/cosyvoice2/bin/python"

# ─── Parse Arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topk)          TOPK_RATIO="$2"; shift 2 ;;
        --target-ckpt)   TARGET_CKPT="$2"; shift 2 ;;
        --ref-ckpt)      REF_CKPT="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; DATA_DIR="${OUTPUT_DIR}/data"; shift 2 ;;
        --audio-dir)     AUDIO_DIR="$2"; shift 2 ;;
        --config)        CONFIG="$2"; shift 2 ;;
        --workers)       NUM_WORKERS="$2"; shift 2 ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Print Config ────────────────────────────────────────────────────────────

echo "============================================================"
echo "Phase 2: Semi-offline Token Scoring"
echo "============================================================"
echo ""
echo "  Target ckpt:  ${TARGET_CKPT}"
echo "  Ref ckpt:     ${REF_CKPT}"
echo "  Config:       ${CONFIG}"
echo "  Top-k ratio:  ${TOPK_RATIO}"
echo "  Device:       ${DEVICE}"
echo "  Output:       ${OUTPUT_DIR}"
echo ""

# ─── Validate ────────────────────────────────────────────────────────────────

for f in "${TARGET_CKPT}" "${REF_CKPT}" "${CONFIG}" "${TRAIN_MANIFEST}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: File not found: ${f}"
        exit 1
    fi
done

if [ ! -d "${QWEN_PRETRAIN}" ]; then
    echo "ERROR: Qwen pretrain not found: ${QWEN_PRETRAIN}"
    exit 1
fi

# ─── Step 1: Prepare D_train parquet (if not already done) ───────────────────

TRAIN_DATA_LIST="${DATA_DIR}/train.data.list"

if [ ! -f "${TRAIN_DATA_LIST}" ]; then
    echo "------------------------------------------------------------"
    echo "Step 1: Preparing D_train parquet data"
    echo "------------------------------------------------------------"
    cd "${PROJECT_ROOT}"
    ${CONDA_PYTHON} -m phase1.prepare_data \
        --ref-manifest "${TRAIN_MANIFEST}" \
        --audio-dir "${AUDIO_DIR}" \
        --output-dir "${DATA_DIR}" \
        --val-ratio 0.0 \
        --seed 42
    # With val-ratio=0, all data goes to train split
    echo ""
else
    echo "D_train parquet already prepared: ${TRAIN_DATA_LIST}"
fi

# ─── Step 2: Run scoring ─────────────────────────────────────────────────────

echo "------------------------------------------------------------"
echo "Step 2: Running dual-model scoring"
echo "------------------------------------------------------------"

export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"
${CONDA_PYTHON} -m phase2.score_tokens \
    --train-data-list "${TRAIN_DATA_LIST}" \
    --target-ckpt "${TARGET_CKPT}" \
    --ref-ckpt "${REF_CKPT}" \
    --config "${CONFIG}" \
    --qwen-pretrain-path "${QWEN_PRETRAIN}" \
    --onnx-path "${ONNX_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --topk-ratio "${TOPK_RATIO}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch "${PREFETCH}"

echo ""
echo "============================================================"
echo "Phase 2 COMPLETE"
echo "============================================================"
echo "  Scores: ${OUTPUT_DIR}/token_scores.pt"
echo "  Summary: ${OUTPUT_DIR}/phase2_summary.json"
