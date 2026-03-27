#!/bin/bash
# Download CosyVoice2-0.5B pretrained model from HuggingFace/ModelScope.
#
# Usage:
#   bash scripts/download_model.sh [DEST_DIR]
#
# Default destination: pretrained_models/CosyVoice2-0.5B

set -euo pipefail

CONDA_ENV="/data/zhenghao/miniforge3/envs/cosyvoice2/bin"
PYTHON="${CONDA_ENV}/python"

DEST_DIR="${1:-/data/zhenghao/repos/token-level/pretrained_models/CosyVoice2-0.5B}"
mkdir -p "${DEST_DIR}"

echo "========================================="
echo "Downloading CosyVoice2-0.5B to: ${DEST_DIR}"
echo "========================================="

# Try HuggingFace first
${PYTHON} -c "
from huggingface_hub import snapshot_download
import os
dest = '${DEST_DIR}'
print(f'Downloading to {dest} ...')
snapshot_download(
    repo_id='FunAudioLLM/CosyVoice2-0.5B',
    local_dir=dest,
    local_dir_use_symlinks=False,
)
print('Download complete!')
"

echo ""
echo "Verifying downloaded files..."
for f in llm.pt flow.pt hift.pt campplus.onnx; do
    if [ -f "${DEST_DIR}/${f}" ]; then
        echo "  ✓ ${f}"
    else
        echo "  ✗ ${f} MISSING"
    fi
done

# Check for Qwen pretrain dir
if [ -d "${DEST_DIR}/CosyVoice-BlankEN" ]; then
    echo "  ✓ CosyVoice-BlankEN/"
else
    echo "  ✗ CosyVoice-BlankEN/ MISSING"
fi

# Check speech tokenizer (could be v2 or v3)
found_tokenizer=0
for f in speech_tokenizer_v2.onnx speech_tokenizer_v3.onnx speech_tokenizer_v2.batch.onnx; do
    if [ -f "${DEST_DIR}/${f}" ]; then
        echo "  ✓ ${f}"
        found_tokenizer=1
    fi
done
if [ $found_tokenizer -eq 0 ]; then
    echo "  ✗ No speech_tokenizer onnx found"
fi

echo ""
echo "Done. Model saved to: ${DEST_DIR}"
