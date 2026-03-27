#!/bin/bash
# =============================================================================
# Phase 1: Train Reference Model (RM) on D_ref
#
# This script orchestrates the full Phase 1 pipeline:
#   1. Prepare parquet data from D_ref manifest
#   2. Generate Phase 1 training config (patched from CosyVoice2 base config)
#   3. Launch CosyVoice2 LLM training via torchrun
#   4. Export frozen RM checkpoint
#
# Prerequisites:
#   - CosyVoice2-0.5B pretrained model downloaded
#     (run: bash scripts/download_model.sh)
#   - Phase 0 completed (phase0_manifest_ref.jsonl exists)
#   - cosyvoice2 conda env activated
#
# Usage:
#   # Dry run (prepare data + config only, no training):
#   bash scripts/phase1_train.sh
#
#   # Full training:
#   bash scripts/phase1_train.sh --run
#
#   # Custom settings:
#   bash scripts/phase1_train.sh --run --gpus 2 --epoch 5 --lr 5e-6
# =============================================================================

set -euo pipefail

# ─── Default Configuration ───────────────────────────────────────────────────

# Project paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COSYVOICE_ROOT="/data/zhenghao/repos/CosyVoice"
PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_models/CosyVoice2-0.5B"

# Data paths
REF_MANIFEST="${PROJECT_ROOT}/phase0_outputs/phase0_manifest_ref.jsonl"
AUDIO_DIR="/data/zhenghao/data/crema_data_1000/audios"

# Output paths
OUTPUT_DIR="${PROJECT_ROOT}/phase1_outputs"
DATA_DIR="${OUTPUT_DIR}/data"
MODEL_DIR="${OUTPUT_DIR}/checkpoints"
TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
RM_DIR="${OUTPUT_DIR}/rm"

# Training hyperparameters
LEARNING_RATE="1e-5"
MAX_EPOCH=3
LOG_INTERVAL=20
SAVE_PER_STEP=-1  # -1 means save per epoch only
ACCUM_GRAD=2
GRAD_CLIP=5
SCHEDULER="constantlr"
WARMUP_STEPS=100

# Resource settings
NUM_GPUS=1
NUM_WORKERS=2
PREFETCH=100
USE_AMP=true
TRAIN_ENGINE="torch_ddp"

# Data preparation
VAL_RATIO=0.05
SEED=42
UTTS_PER_SHARD=500

# Control flags
RUN_TRAINING=false

# Conda env
CONDA_PYTHON="/data/zhenghao/miniforge3/envs/cosyvoice2/bin/python"
CONDA_TORCHRUN="/data/zhenghao/miniforge3/envs/cosyvoice2/bin/torchrun"

# ─── Parse Arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)           RUN_TRAINING=true; shift ;;
        --gpus)          NUM_GPUS="$2"; shift 2 ;;
        --epoch)         MAX_EPOCH="$2"; shift 2 ;;
        --lr)            LEARNING_RATE="$2"; shift 2 ;;
        --accum-grad)    ACCUM_GRAD="$2"; shift 2 ;;
        --save-per-step) SAVE_PER_STEP="$2"; shift 2 ;;
        --ref-manifest)  REF_MANIFEST="$2"; shift 2 ;;
        --audio-dir)     AUDIO_DIR="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; DATA_DIR="${OUTPUT_DIR}/data"; MODEL_DIR="${OUTPUT_DIR}/checkpoints"; TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"; RM_DIR="${OUTPUT_DIR}/rm"; shift 2 ;;
        --pretrained)    PRETRAINED_MODEL_DIR="$2"; shift 2 ;;
        --cosyvoice-root) COSYVOICE_ROOT="$2"; shift 2 ;;
        --train-engine)  TRAIN_ENGINE="$2"; shift 2 ;;
        --no-amp)        USE_AMP=false; shift ;;
        --workers)       NUM_WORKERS="$2"; shift 2 ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────────────

echo "============================================================"
echo "Phase 1: Reference Model Training"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Project root:      ${PROJECT_ROOT}"
echo "  CosyVoice root:    ${COSYVOICE_ROOT}"
echo "  Pretrained model:  ${PRETRAINED_MODEL_DIR}"
echo "  Ref manifest:      ${REF_MANIFEST}"
echo "  Audio dir:         ${AUDIO_DIR}"
echo "  Output dir:        ${OUTPUT_DIR}"
echo "  LR:                ${LEARNING_RATE}"
echo "  Max epoch:         ${MAX_EPOCH}"
echo "  GPUs:              ${NUM_GPUS}"
echo "  Train engine:      ${TRAIN_ENGINE}"
echo "  AMP:               ${USE_AMP}"
echo "  Run training:      ${RUN_TRAINING}"
echo ""

# Check prerequisites
if [ ! -f "${REF_MANIFEST}" ]; then
    echo "ERROR: Ref manifest not found: ${REF_MANIFEST}"
    echo "Please run Phase 0 first."
    exit 1
fi

if [ ! -d "${AUDIO_DIR}" ]; then
    echo "ERROR: Audio directory not found: ${AUDIO_DIR}"
    exit 1
fi

COSYVOICE_TRAIN_PY="${COSYVOICE_ROOT}/cosyvoice/bin/train.py"
if [ ! -f "${COSYVOICE_TRAIN_PY}" ]; then
    echo "ERROR: CosyVoice train.py not found: ${COSYVOICE_TRAIN_PY}"
    exit 1
fi

BASE_CONFIG="${COSYVOICE_ROOT}/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml"
if [ ! -f "${BASE_CONFIG}" ]; then
    echo "ERROR: Base config not found: ${BASE_CONFIG}"
    exit 1
fi

QWEN_PRETRAIN="${PRETRAINED_MODEL_DIR}/CosyVoice-BlankEN"
INIT_CHECKPOINT="${PRETRAINED_MODEL_DIR}/llm.pt"
ONNX_PATH="${PRETRAINED_MODEL_DIR}"

if [ "${RUN_TRAINING}" = true ]; then
    if [ ! -d "${QWEN_PRETRAIN}" ]; then
        echo "ERROR: Qwen pretrain path not found: ${QWEN_PRETRAIN}"
        echo "Please run: bash scripts/download_model.sh"
        exit 1
    fi
    if [ ! -f "${INIT_CHECKPOINT}" ]; then
        echo "WARNING: Init checkpoint not found: ${INIT_CHECKPOINT}"
        echo "Training will start from scratch (Qwen pretrain only)."
        INIT_CHECKPOINT=""
    fi
fi

# ─── Step 1: Prepare Data ────────────────────────────────────────────────────

echo "------------------------------------------------------------"
echo "Step 1: Preparing parquet data from D_ref"
echo "------------------------------------------------------------"

${CONDA_PYTHON} -m phase1.prepare_data \
    --ref-manifest "${REF_MANIFEST}" \
    --audio-dir "${AUDIO_DIR}" \
    --output-dir "${DATA_DIR}" \
    --val-ratio ${VAL_RATIO} \
    --seed ${SEED} \
    --utts-per-shard ${UTTS_PER_SHARD}

TRAIN_DATA_LIST="${DATA_DIR}/train.data.list"
CV_DATA_LIST="${DATA_DIR}/cv.data.list"

if [ ! -f "${TRAIN_DATA_LIST}" ] || [ ! -f "${CV_DATA_LIST}" ]; then
    echo "ERROR: Data preparation failed!"
    exit 1
fi
echo "Data prepared successfully."
echo ""

# ─── Step 2: Generate Phase 1 Config ─────────────────────────────────────────

echo "------------------------------------------------------------"
echo "Step 2: Generating Phase 1 training config"
echo "------------------------------------------------------------"

PHASE1_CONFIG="${OUTPUT_DIR}/phase1_cosyvoice2.yaml"
mkdir -p "${OUTPUT_DIR}"

${CONDA_PYTHON} << CONFEOF
import re

with open("${BASE_CONFIG}", "r") as f:
    lines = f.readlines()

# Patch train_conf values
in_train_conf = False
in_optim_conf = False
in_scheduler_conf = False

for i, line in enumerate(lines):
    stripped = line.strip()
    # Detect top-level keys
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

    # Detect optim_conf / scheduler_conf sub-blocks
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

    # Patch lr
    if in_optim_conf and re.match(r"^\s{8}lr:", line):
        lines[i] = "        lr: ${LEARNING_RATE}\n"
    # Patch scheduler_conf warmup_steps
    if in_scheduler_conf and re.match(r"^\s{8}warmup_steps:", line):
        lines[i] = "        warmup_steps: ${WARMUP_STEPS}\n"
    # Patch top-level train_conf values
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

with open("${PHASE1_CONFIG}", "w") as f:
    f.writelines(lines)

print("Phase 1 config written to: ${PHASE1_CONFIG}")
CONFEOF

echo "Config generated: ${PHASE1_CONFIG}"
echo ""

# ─── Step 3: Build Training Command ──────────────────────────────────────────

echo "------------------------------------------------------------"
echo "Step 3: Building training command"
echo "------------------------------------------------------------"

# Build torchrun command
TRAIN_CMD="${CONDA_TORCHRUN} --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS}"
TRAIN_CMD="${TRAIN_CMD} ${COSYVOICE_TRAIN_PY}"
TRAIN_CMD="${TRAIN_CMD} --train_engine ${TRAIN_ENGINE}"
TRAIN_CMD="${TRAIN_CMD} --model llm"
TRAIN_CMD="${TRAIN_CMD} --config ${PHASE1_CONFIG}"
TRAIN_CMD="${TRAIN_CMD} --train_data ${TRAIN_DATA_LIST}"
TRAIN_CMD="${TRAIN_CMD} --cv_data ${CV_DATA_LIST}"
TRAIN_CMD="${TRAIN_CMD} --qwen_pretrain_path ${QWEN_PRETRAIN}"
TRAIN_CMD="${TRAIN_CMD} --onnx_path ${ONNX_PATH}"
TRAIN_CMD="${TRAIN_CMD} --model_dir ${MODEL_DIR}"
TRAIN_CMD="${TRAIN_CMD} --tensorboard_dir ${TENSORBOARD_DIR}"
TRAIN_CMD="${TRAIN_CMD} --ddp.dist_backend nccl"
TRAIN_CMD="${TRAIN_CMD} --num_workers ${NUM_WORKERS}"
TRAIN_CMD="${TRAIN_CMD} --prefetch ${PREFETCH}"
TRAIN_CMD="${TRAIN_CMD} --pin_memory"

if [ "${USE_AMP}" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --use_amp"
fi

if [ -n "${INIT_CHECKPOINT}" ] && [ -f "${INIT_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --checkpoint ${INIT_CHECKPOINT}"
fi

if [ "${TRAIN_ENGINE}" = "deepspeed" ]; then
    DS_CONFIG="${COSYVOICE_ROOT}/examples/libritts/cosyvoice2/conf/ds_stage2.json"
    TRAIN_CMD="${TRAIN_CMD} --deepspeed_config ${DS_CONFIG}"
    TRAIN_CMD="${TRAIN_CMD} --deepspeed.save_states model+optimizer"
fi

# Save command for reference
CMD_FILE="${OUTPUT_DIR}/phase1_train_command.sh"
cat > "${CMD_FILE}" << CMD_INNER
#!/bin/bash
# Phase 1 training command (auto-generated)
# To re-run training directly:
#   cd ${COSYVOICE_ROOT} && bash ${CMD_FILE}

export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:\${PYTHONPATH:-}"

${TRAIN_CMD}
CMD_INNER
chmod +x "${CMD_FILE}"

echo "Training command saved to: ${CMD_FILE}"
echo ""
echo "Command:"
echo "  ${TRAIN_CMD}"
echo ""

# ─── Step 4: Run Training (if requested) ─────────────────────────────────────

if [ "${RUN_TRAINING}" = true ]; then
    echo "------------------------------------------------------------"
    echo "Step 4: Launching training"
    echo "------------------------------------------------------------"
    echo ""

    mkdir -p "${MODEL_DIR}" "${TENSORBOARD_DIR}"

    export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
    cd "${COSYVOICE_ROOT}"

    echo "Starting training at $(date)"
    echo ""

    ${TRAIN_CMD}
    TRAIN_EXIT=$?

    echo ""
    echo "Training finished at $(date) with exit code: ${TRAIN_EXIT}"

    if [ ${TRAIN_EXIT} -eq 0 ]; then
        echo ""
        echo "------------------------------------------------------------"
        echo "Step 5: Exporting frozen RM checkpoint"
        echo "------------------------------------------------------------"

        cd "${PROJECT_ROOT}"
        ${CONDA_PYTHON} -m phase1.export_checkpoint \
            --model-dir "${MODEL_DIR}" \
            --output-dir "${RM_DIR}"

        echo ""
        echo "============================================================"
        echo "Phase 1 COMPLETE"
        echo "============================================================"
        echo "  Checkpoints: ${MODEL_DIR}"
        echo "  Frozen RM:   ${RM_DIR}"
        echo "  TensorBoard: ${TENSORBOARD_DIR}"
        echo "  View logs:   tensorboard --logdir ${TENSORBOARD_DIR}"
    else
        echo ""
        echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
        exit ${TRAIN_EXIT}
    fi
else
    echo "============================================================"
    echo "Phase 1 DRY RUN COMPLETE"
    echo "============================================================"
    echo ""
    echo "Data and config are prepared. To launch training:"
    echo ""
    echo "  # Option 1: Re-run this script with --run"
    echo "  bash scripts/phase1_train.sh --run"
    echo ""
    echo "  # Option 2: Run the generated command directly"
    echo "  cd ${COSYVOICE_ROOT}"
    echo "  export PYTHONPATH=${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS"
    echo "  bash ${CMD_FILE}"
    echo ""
    echo "  # Option 3: After training, export RM manually"
    echo "  ${CONDA_PYTHON} -m phase1.export_checkpoint --model-dir ${MODEL_DIR} --output-dir ${RM_DIR}"
    echo ""
fi
