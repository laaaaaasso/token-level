#!/bin/bash
# ===========================================================================
# Phase 1: RM 训练启动脚本 (训练集群用)
#
# 前置条件 (在debug集群完成):
#   bash stage_e/prepare_phase1_data.sh
#
# 使用:
#   cd /data/zhenghao/repos/token-level && bash stage_e/launch_phase1_30h.sh
#   cd /data/zhenghao/repos/token-level && bash stage_e/launch_phase1_30h.sh --gpus 2
#
# 训练规模: D_ref 4169样本, CV 3268样本, 3 epochs, ~6255步, ~1-2h (单卡H200)
# ===========================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COSYVOICE_ROOT="/data/zhenghao/repos/CosyVoice"
CONDA_PYTHON="/data/zhenghao/miniforge3/envs/cosyvoice2/bin/python"
CONDA_TORCHRUN="/data/zhenghao/miniforge3/envs/cosyvoice2/bin/torchrun"

OUTPUT_DIR="${PROJECT_ROOT}/phase1_outputs_30h"
PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_models/CosyVoice2-0.5B"

# 训练超参
NUM_GPUS=1
USE_AMP=true
TRAIN_ENGINE="torch_ddp"
NUM_WORKERS=4
PREFETCH=100

# 命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *)            echo "Usage: bash $0 [--gpus N] [--output-dir DIR]"; exit 1 ;;
    esac
done

# 衍生路径
DATA_DIR="${OUTPUT_DIR}/data"
MODEL_DIR="${OUTPUT_DIR}/checkpoints"
TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
RM_DIR="${OUTPUT_DIR}/rm"
LOG_FILE="${OUTPUT_DIR}/phase1_train.log"
PHASE1_CONFIG="${OUTPUT_DIR}/phase1_cosyvoice2.yaml"
TRAIN_DATA_LIST="${DATA_DIR}/train.data.list"
CV_DATA_LIST="${DATA_DIR}/cv.data.list"

# ── 前置检查 ──────────────────────────────────────────────────────────

echo "Phase 1 RM Training"
echo ""

ERRORS=0
for f in "${TRAIN_DATA_LIST}" "${CV_DATA_LIST}" "${PHASE1_CONFIG}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found. Run prepare_phase1_data.sh first."
        ERRORS=$((ERRORS+1))
    fi
done
for d in "${COSYVOICE_ROOT}" "${PRETRAINED_MODEL_DIR}/CosyVoice-BlankEN"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: $d not found."
        ERRORS=$((ERRORS+1))
    fi
done
[ ${ERRORS} -gt 0 ] && exit 1

mkdir -p "${MODEL_DIR}" "${TENSORBOARD_DIR}" "${RM_DIR}"

# GPU 检查
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "GPU: ${GPU_COUNT}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    [ "${NUM_GPUS}" -gt "${GPU_COUNT}" ] && NUM_GPUS=${GPU_COUNT}
fi

echo "Config:  ${PHASE1_CONFIG}"
echo "Train:   ${TRAIN_DATA_LIST} ($(wc -l < "${TRAIN_DATA_LIST}") shards)"
echo "CV:      ${CV_DATA_LIST} ($(wc -l < "${CV_DATA_LIST}") shards)"
echo "GPUs:    ${NUM_GPUS}"
echo "Output:  ${OUTPUT_DIR}"
echo ""

# ── 构建训练命令 ──────────────────────────────────────────────────────

TRAIN_CMD="${CONDA_TORCHRUN} --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS}"
TRAIN_CMD="${TRAIN_CMD} ${COSYVOICE_ROOT}/cosyvoice/bin/train.py"
TRAIN_CMD="${TRAIN_CMD} --train_engine ${TRAIN_ENGINE}"
TRAIN_CMD="${TRAIN_CMD} --model llm"
TRAIN_CMD="${TRAIN_CMD} --config ${PHASE1_CONFIG}"
TRAIN_CMD="${TRAIN_CMD} --train_data ${TRAIN_DATA_LIST}"
TRAIN_CMD="${TRAIN_CMD} --cv_data ${CV_DATA_LIST}"
TRAIN_CMD="${TRAIN_CMD} --qwen_pretrain_path ${PRETRAINED_MODEL_DIR}/CosyVoice-BlankEN"
TRAIN_CMD="${TRAIN_CMD} --onnx_path ${PRETRAINED_MODEL_DIR}"
TRAIN_CMD="${TRAIN_CMD} --model_dir ${MODEL_DIR}"
TRAIN_CMD="${TRAIN_CMD} --tensorboard_dir ${TENSORBOARD_DIR}"
TRAIN_CMD="${TRAIN_CMD} --ddp.dist_backend nccl"
TRAIN_CMD="${TRAIN_CMD} --num_workers ${NUM_WORKERS}"
TRAIN_CMD="${TRAIN_CMD} --prefetch ${PREFETCH}"
TRAIN_CMD="${TRAIN_CMD} --pin_memory"

[ "${USE_AMP}" = true ] && TRAIN_CMD="${TRAIN_CMD} --use_amp"
[ -f "${PRETRAINED_MODEL_DIR}/llm.pt" ] && TRAIN_CMD="${TRAIN_CMD} --checkpoint ${PRETRAINED_MODEL_DIR}/llm.pt"

# ── 训练 ──────────────────────────────────────────────────────────────

export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

echo "Training started: $(date)"
cd "${COSYVOICE_ROOT}"
${TRAIN_CMD} 2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT=${PIPESTATUS[0]}
cd "${PROJECT_ROOT}"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "ERROR: Training failed (exit ${TRAIN_EXIT}). See ${LOG_FILE}"
    exit ${TRAIN_EXIT}
fi
echo "Training finished: $(date)"

# ── 导出 RM ──────────────────────────────────────────────────────────

echo ""
echo "Exporting frozen RM..."
${CONDA_PYTHON} -m phase1.export_checkpoint \
    --model-dir "${MODEL_DIR}" \
    --output-dir "${RM_DIR}"

if [ ! -f "${RM_DIR}/rm_frozen.pt" ]; then
    echo "ERROR: rm_frozen.pt not generated!"
    exit 1
fi

echo ""
echo "Done. Outputs:"
echo "  Checkpoints:  ${MODEL_DIR}/"
echo "  Frozen RM:    ${RM_DIR}/rm_frozen.pt"
echo "  TensorBoard:  ${TENSORBOARD_DIR}/"
echo "  Log:          ${LOG_FILE}"
