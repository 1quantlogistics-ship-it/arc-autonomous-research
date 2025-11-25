#!/bin/bash
# Launch distributed training on RunPod
# Phase F - Infrastructure & Stability Track

set -e

# Detect GPU count
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "Detected $GPU_COUNT GPUs"

# Parse arguments
SCRIPT="${1:-train.py}"
shift || true

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Launching distributed training on $GPU_COUNT GPUs..."
    torchrun \
        --nproc_per_node=$GPU_COUNT \
        --master_port=29500 \
        "$SCRIPT" "$@"
else
    echo "Single GPU mode..."
    python "$SCRIPT" "$@"
fi
