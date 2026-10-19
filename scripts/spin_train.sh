#!/bin/bash
# Train one SPIN run on one GPU; flags go to training.spin.train and name the run.
#   bash scripts/spin_train.sh --encoder qwen --layer 18 --m 5 --data full
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH=$PWD
RUN=${RUN:-}                       # job-submission prefix for every GPU command, e.g. RUN="srun --gres=gpu:1"
venv() { source "$(dirname "$(dirname "$(python3 -c "from multi3ir.paths import env_python; print(env_python(\"$1\"))")")")/bin/activate"; }
ENC=qwen; prev=""; for a in "$@"; do [ "$prev" = --encoder ] && ENC=$a; prev=$a; done
venv "$ENC"
NAME=$(python -c 'import sys; from multi3ir.training.spin import train as T; p = T.build_parser(); print(T.run_name(p.parse_args(sys.argv[1:]), p))' "$@")
LOG=results/logs/training/$NAME.log; mkdir -p results/logs/training
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d-%H%M%S)"     # keep every attempt's log
echo "[train] $NAME -> $LOG"
$RUN python -u -m multi3ir.training.spin.train "$@" 2>&1 | tee "$LOG"
python -m multi3ir.training.watch_loss "$LOG" --once             # loss curve -> ${LOG%.log}.loss.png
echo "[train] done: results/ckpts/spin/$NAME/spin_best.pt"
