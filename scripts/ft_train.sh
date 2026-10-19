#!/bin/bash
# Naive fine-tuning baseline (training.baselines.finetune) with torchrun over GPUS GPUs.
#   bash scripts/ft_train.sh qwen_ft_full
#   BACKEND=mmembed MODE=lora LR=1e-6 TAU=0.02 STEPS=400 bash scripts/ft_train.sh mmembed_ft_lora
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH=$PWD
RUN=${RUN:-}                       # job-submission prefix for every GPU command, e.g. RUN="srun --gres=gpu:1"
venv() { source "$(dirname "$(dirname "$(python3 -c "from multi3ir.paths import env_python; print(env_python(\"$1\"))")")")/bin/activate"; }
TAG=${1:?usage: ft_train.sh TAG}
BACKEND=${BACKEND:-qwen}; MODE=${MODE:-full}; LR=${LR:-1e-5}; TAU=${TAU:-0.05}; STEPS=${STEPS:-400}; BATCH=${BATCH:-16}
GPUS=${GPUS:-4}; VAL_EVERY=${VAL_EVERY:-20}; TRAIN_CORPUS=${TRAIN_CORPUS:-gold}; VAL_CORPUS=${VAL_CORPUS:-bench}
venv "$BACKEND"
OUT=results/ckpts/$TAG; LOG=results/logs/training/$TAG.log; mkdir -p "$OUT" results/logs/training
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d-%H%M%S)"
$RUN python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" -m multi3ir.training.baselines.finetune \
    --backend "$BACKEND" --mode "$MODE" --out "$OUT" --lr "$LR" --tau "$TAU" --num-steps "$STEPS" --batch-size "$BATCH" \
    --val-every "$VAL_EVERY" --train-corpus "$TRAIN_CORPUS" --val-corpus "$VAL_CORPUS" 2>&1 | tee "$LOG"
python -m multi3ir.training.watch_loss "$LOG" --once
echo "[ft] done: $OUT"
