#!/bin/bash
# =============================================================================
# Phase 3-B1: Full SLM Training via phase3.train (disable_mask)
#
# Baseline: reuse Phase 3 selective pipeline (phase3.train) so that
# TensorBoard scalars和 logging 完全一致，但通过 --disable_mask
# 让所有 speech token 参与 SLM loss，相当于 full-token 训练。
#
# Usage:
#   bash scripts/phase3_full_via_phase3_train.sh           # dry run
#   bash scripts/phase3_full_via_phase3_train.sh --run     # actual training
# =============================================================================

set -euo pipefail

# ─── Default Configuration ───────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COSYVOICE_ROOT="/data/zhenghao/repos/CosyVoice"
PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_models/CosyVoice2-0.5B"

# Model paths
CONFIG="${PROJECT_ROOT}/phase1_outputs/phase1_cosyvoice2.yaml"
QWEN_PRETRAIN="${PRETRAINED_MODEL_DIR}/CosyVoice-BlankEN"
ONNX_PATH="${PRETRAINED_MODEL_DIR}"
INIT_CHECKPOINT="${PROJECT_ROOT}/phase1_outputs/rm/rm_frozen.pt"

# Data paths（与 selective 一致）
TRAIN_DATA_LIST="${PROJECT_ROOT}/phase2_outputs/data/train.data.list"
CV_DATA_LIST="${PROJECT_ROOT}/phase1_outputs/data/cv.data.list"
MASK_PATH="${PROJECT_ROOT}/phase2_outputs/token_scores.pt"

# Output paths（单独目录，便于和 selective 区分）
OUTPUT_DIR="${PROJECT_ROOT}/phase3_full2_outputs"
MODEL_DIR="${OUTPUT_DIR}/checkpoints"
TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"

# Training hyperparameters（对齐 selective）
LEARNING_RATE="5e-6"
MAX_EPOCH=3
LOG_INTERVAL=20
SAVE_PER_STEP=-1
ACCUM_GRAD=2
GRAD_CLIP=5
SCHEDULER="constantlr"
WARMUP_STEPS=50

# Resource settings
NUM_GPUS=1
NUM_WORKERS=2
PREFETCH=100
USE_AMP=true
TRAIN_ENGINE="torch_ddp"

# Control
RUN_TRAINING=false

# Conda
CONDA_TORCHRUN="/data/zhenghao/miniforge3/envs/cosyvoice2/bin/torchrun"

# ─── Parse Arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)           RUN_TRAINING=true; shift ;;
        --gpus)          NUM_GPUS="$2"; shift 2 ;;
        --epoch)         MAX_EPOCH="$2"; shift 2 ;;
        --lr)            LEARNING_RATE="$2"; shift 2 ;;
        --accum-grad)    ACCUM_GRAD="$2"; shift 2 ;;
        --mask-path)     MASK_PATH="$2"; shift 2 ;;
        --checkpoint)    INIT_CHECKPOINT="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; MODEL_DIR="${OUTPUT_DIR}/checkpoints"; TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"; shift 2 ;;
        --config)        CONFIG="$2"; shift 2 ;;
        --no-amp)        USE_AMP=false; shift ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────────────

echo "============================================================"
echo "Phase 3-B1: Full SLM via phase3.train (--disable_mask)"
echo "============================================================"
echo ""
echo "  Config:          ${CONFIG}"
echo "  Init checkpoint: ${INIT_CHECKPOINT}"
echo "  Mask path (unused when disable_mask): ${MASK_PATH}"
echo "  Train data:      ${TRAIN_DATA_LIST}"
echo "  CV data:         ${CV_DATA_LIST}"
echo "  LR:              ${LEARNING_RATE}"
echo "  Max epoch:       ${MAX_EPOCH}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  AMP:             ${USE_AMP}"
echo "  Run training:    ${RUN_TRAINING}"
echo ""

for f in "${CONFIG}" "${TRAIN_DATA_LIST}" "${CV_DATA_LIST}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: File not found: ${f}"
        exit 1
    fi
done

if [ ! -d "${QWEN_PRETRAIN}" ]; then
    echo "ERROR: Qwen pretrain not found: ${QWEN_PRETRAIN}"
    exit 1
fi

# ─── Generate config（复用 Phase 1 配置，只 patch 训练超参） ────────────────

PHASE3_CONFIG="${OUTPUT_DIR}/phase3_full_cosyvoice2.yaml"
mkdir -p "${OUTPUT_DIR}" "${MODEL_DIR}" "${TENSORBOARD_DIR}"

/data/zhenghao/miniforge3/envs/cosyvoice2/bin/python << CONFEOF
import re

with open("${CONFIG}", "r") as f:
    lines = f.readlines()

in_train_conf = False
in_optim_conf = False
in_scheduler_conf = False

for i, line in enumerate(lines):
    stripped = line.strip()
    if line and not line[0].isspace() and ":" in line:
        if stripped.startswith("train_conf:"):
            in_train_conf = True
            in_optim_conf = False
            in_scheduler_conf = False
            continue
        elif in_train_conf:
            in_train_conf = False
            in_optim_conf = False
            in_scheduler_conf = False

    if not in_train_conf:
        continue

    if re.match(r"^\s{4}optim_conf:\s*$", line):
        in_optim_conf = True
        in_scheduler_conf = False
        continue
    if re.match(r"^\s{4}scheduler_conf:\s*$", line):
        in_scheduler_conf = True
        in_optim_conf = False
        continue
    if re.match(r"^\s{4}[a-zA-Z_]\w*:", line) and not re.match(r"^\s{4}(optim_conf|scheduler_conf):", line):
        in_optim_conf = False
        in_scheduler_conf = False

    if in_optim_conf and re.match(r"^\s{8}lr:", line):
        lines[i] = "        lr: ${LEARNING_RATE}\n"
    if in_scheduler_conf and re.match(r"^\s{8}warmup_steps:", line):
        lines[i] = "        warmup_steps: ${WARMUP_STEPS}\n"
    if re.match(r"^\s{4}max_epoch:", line):
        lines[i] = "    max_epoch: ${MAX_EPOCH}\n"
    if re.match(r"^\s{4}log_interval:", line):
        lines[i] = "    log_interval: ${LOG_INTERVAL}\n"
    if re.match(r"^\s{4}save_per_step:", line):
        lines[i] = "    save_per_step: ${SAVE_PER_STEP}\n"
    if re.match(r"^\s{4}accum_grad:", line):
        lines[i] = "    accum_grad: ${ACCUM_GRAD}\n"
    if re.match(r"^\s{4}grad_clip:", line):
        lines[i] = "    grad_clip: ${GRAD_CLIP}\n"
    if re.match(r"^\s{4}scheduler:", line):
        lines[i] = "    scheduler: ${SCHEDULER}\n"

with open("${PHASE3_CONFIG}", "w") as f:
    f.writelines(lines)

print("Phase 3 full-training config written to: ${PHASE3_CONFIG}")
CONFEOF

echo "Config: ${PHASE3_CONFIG}"
echo ""

# ─── Build training command（调用 phase3.train，开启 --disable_mask） ───────

TRAIN_CMD="${CONDA_TORCHRUN} --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS}"
TRAIN_CMD="${TRAIN_CMD} -m phase3.train"
TRAIN_CMD="${TRAIN_CMD} --train_engine ${TRAIN_ENGINE}"
TRAIN_CMD="${TRAIN_CMD} --config ${PHASE3_CONFIG}"
TRAIN_CMD="${TRAIN_CMD} --train_data ${TRAIN_DATA_LIST}"
TRAIN_CMD="${TRAIN_CMD} --cv_data ${CV_DATA_LIST}"
TRAIN_CMD="${TRAIN_CMD} --qwen_pretrain_path ${QWEN_PRETRAIN}"
TRAIN_CMD="${TRAIN_CMD} --onnx_path ${ONNX_PATH}"
TRAIN_CMD="${TRAIN_CMD} --mask_path ${MASK_PATH}"
TRAIN_CMD="${TRAIN_CMD} --model_dir ${MODEL_DIR}"
TRAIN_CMD="${TRAIN_CMD} --tensorboard_dir ${TENSORBOARD_DIR}"
TRAIN_CMD="${TRAIN_CMD} --ddp.dist_backend nccl"
TRAIN_CMD="${TRAIN_CMD} --num_workers ${NUM_WORKERS}"
TRAIN_CMD="${TRAIN_CMD} --prefetch ${PREFETCH}"
TRAIN_CMD="${TRAIN_CMD} --pin_memory"
TRAIN_CMD="${TRAIN_CMD} --disable_mask"

if [ "${USE_AMP}" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_amp"
fi

if [ -n "${INIT_CHECKPOINT}" ] && [ -f "${INIT_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --checkpoint ${INIT_CHECKPOINT}"
fi

# Save command
CMD_FILE="${OUTPUT_DIR}/phase3_full_train_command.sh"
cat > "${CMD_FILE}" << CMD_INNER
#!/bin/bash
# Phase 3-B1 full-training (via phase3.train) command (auto-generated)
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PROJECT_ROOT}:\${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
${TRAIN_CMD}
CMD_INNER
chmod +x "${CMD_FILE}"

echo "Command saved to: ${CMD_FILE}"
echo ""
echo "  ${TRAIN_CMD}"
echo ""

# ─── Run training ────────────────────────────────────────────────────────────

if [ "${RUN_TRAINING}" = true ]; then
    echo "------------------------------------------------------------"
    echo "Launching Phase 3-B1 full SLM training via phase3.train (--disable_mask)"
    echo "------------------------------------------------------------"

    export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PROJECT_ROOT}:${PYTHONPATH:-}"
    cd "${PROJECT_ROOT}"

    echo "Starting at $(date)"
    ${TRAIN_CMD}
    TRAIN_EXIT=$?

    echo ""
    echo "Training finished at $(date) with exit code: ${TRAIN_EXIT}"

    if [ ${TRAIN_EXIT} -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "Phase 3-B1 FULL (phase3.train) COMPLETE"
        echo "============================================================"
        echo "  Checkpoints:  ${MODEL_DIR}"
        echo "  TensorBoard:  ${TENSORBOARD_DIR}"
        echo "  View logs:    tensorboard --logdir ${TENSORBOARD_DIR}"
    else
        echo "ERROR: Full training (phase3.train) failed with exit code ${TRAIN_EXIT}"
        exit ${TRAIN_EXIT}
    fi
else
    echo "============================================================"
    echo "Phase 3-B1 FULL (phase3.train) DRY RUN COMPLETE"
    echo "============================================================"
    echo ""
    echo "To launch training:"
    echo "  bash scripts/phase3_full_via_phase3_train.sh --run"
    echo ""
    echo "Or run directly:"
    echo "  bash ${CMD_FILE}"
fi

