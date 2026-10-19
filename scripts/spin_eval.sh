#!/bin/bash
# Evaluate SPIN runs on the 1,039-question bench: encode the queries (unless the npz exists), then score.
#   ENCODER=qwen bash scripts/spin_eval.sh RUN [RUN ...]        # MMR=0.2 also reports an MMR merge
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH=$PWD
RUN=${RUN:-}                       # job-submission prefix for every GPU command, e.g. RUN="srun --gres=gpu:1"
venv() { source "$(dirname "$(dirname "$(python3 -c "from multi3ir.paths import env_python; print(env_python(\"$1\"))")")")/bin/activate"; }
ENCODER=${ENCODER:-qwen}; MMR=${MMR:-}
venv "$ENCODER"
EMB=$(python -c "from multi3ir import paths as P; print(P.DOC_EMB['$ENCODER'])")
LOGD=results/logs/eval_1039; mkdir -p "$LOGD" results/bench/query_embedding/spin
for T in "$@"; do
  CK=results/ckpts/spin/$T/spin_best.pt; NPZ=results/bench/query_embedding/spin/$(basename "$EMB")_$T.npz
  LOG=$LOGD/$T.log; [ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d-%H%M%S)"
  [ -f "$NPZ" ] || $RUN python -u -m multi3ir.training.spin.encode --encoder "$ENCODER" --ckpt "$CK" --out "$NPZ" 2>&1 | tee "$LOG"
  $RUN python -u -m multi3ir.eval.run_eval --query-type spin --query-npz "$NPZ" --tag "${T}_1039" ${MMR:+--mmr "$MMR"} \
      --corpus-img "$EMB/corpus_docs.npz" --corpus-txt "$EMB/corpus_docs_text.npz" 2>&1 | tee -a "$LOG"
done
