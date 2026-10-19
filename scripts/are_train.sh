#!/bin/bash
# ARE baseline (training.baselines.are.train), one GPU.
#   bash scripts/are_train.sh 5                                  # K=5, qwen
#   BACKBONE=gme LR=3e-5 TAU=0.03 bash scripts/are_train.sh 5
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH=$PWD
RUN=${RUN:-}                       # job-submission prefix for every GPU command, e.g. RUN="srun --gres=gpu:1"
venv() { source "$(dirname "$(dirname "$(python3 -c "from multi3ir.paths import env_python; print(env_python(\"$1\"))")")")/bin/activate"; }
K=${1:?usage: are_train.sh K}
BACKBONE=${BACKBONE:-qwen}; LR=${LR:-2e-4}; TAU=${TAU:-0.05}; STEPS=${STEPS:-5000}; BSZ=${BSZ:-8}; ACCUM=${ACCUM:-4}; VAL_EVERY=${VAL_EVERY:-0}
venv "$BACKBONE"
TAG=are_${BACKBONE}_K${K}_lr${LR}_tau$TAU; OUT=results/ckpts/$TAG; LOG=results/logs/training/$TAG.log
mkdir -p "$OUT" results/logs/training; [ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d-%H%M%S)"
$RUN python -u -m multi3ir.training.baselines.are.train --out "$OUT" --backbone "$BACKBONE" --K "$K" --lr "$LR" --tau "$TAU" \
    --num-steps "$STEPS" --batch-size "$BSZ" --accum "$ACCUM" --val-every "$VAL_EVERY" 2>&1 | tee "$LOG"
python -m multi3ir.training.watch_loss "$LOG" --once
echo "[are] done: $OUT"
