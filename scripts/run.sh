#!/usr/bin/env bash
set -euo pipefail
EXP_NAME="${1:-islim}"
shift || true
PYTHONPATH="$(dirname "$0")/..:$PYTHONPATH" python scripts/train.py \
    --split posestitch --epochs 200 --batch_size 4 --grad_accum 4 \
    --lr 1e-4 --dropout 0.2 --label_smoothing 0.1 --query_tokens 16 \
    --clip_weight 1.0 --clip_ramp 0.1 --sem_weight 0.05 --sem_ramp 0.005 \
    --struct_weight 0.01 --cool_down 60 --exp_name "$EXP_NAME" "$@"
