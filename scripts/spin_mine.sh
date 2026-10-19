#!/bin/bash
# Query embeddings + one hard-negative table per anchor (persp, question).
#   ENCODER=qwen K=2048 bash scripts/spin_mine.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH=$PWD
RUN=${RUN:-}                       # job-submission prefix for every GPU command, e.g. RUN="srun --gres=gpu:1"
venv() { source "$(dirname "$(dirname "$(python3 -c "from multi3ir.paths import env_python; print(env_python(\"$1\"))")")")/bin/activate"; }
ENCODER=${ENCODER:-qwen}; DATA=${DATA:-full}; K=${K:-2048}
venv "$ENCODER"
M=results/ckpts/spin_mining; Q=$M/${ENCODER}_${DATA}_queries.npz; LOGD=results/logs/mining; mkdir -p "$M" "$LOGD"
[ -f "$Q" ] || $RUN python -u -m multi3ir.training.spin.precompute queries --encoder "$ENCODER" --data "$DATA" --out "$Q" 2>&1 | tee "$LOGD/${ENCODER}_queries.log"
for A in persp question; do
  OUT=$M/${ENCODER}_${DATA}_${A}_k$K.npz; [ -f "$OUT" ] && { echo "[skip] $OUT"; continue; }
  $RUN python -u -m multi3ir.training.spin.mine --encoder "$ENCODER" --data "$DATA" --queries "$Q" --anchor "$A" --k "$K" --out "$OUT" 2>&1 | tee "$LOGD/${ENCODER}_$A.log"
done
echo "[mine] $M"
