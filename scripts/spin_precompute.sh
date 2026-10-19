#!/bin/bash
# Cache the hidden states of the training queries at the injection layer, shard by shard, then repack.
#   BACKEND=qwen LAYER=18 SHARDS=16 bash scripts/spin_precompute.sh          # SHARD_LIST="0 1 2" to run a subset (parallel workers)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH=$PWD
RUN=${RUN:-}                       # job-submission prefix for every GPU command, e.g. RUN="srun --gres=gpu:1"
venv() { source "$(dirname "$(dirname "$(python3 -c "from multi3ir.paths import env_python; print(env_python(\"$1\"))")")")/bin/activate"; }
BACKEND=${BACKEND:-qwen}; LAYER=${LAYER:-18}; SHARDS=${SHARDS:-16}; MAXLEN=${MAXLEN:-256}
venv "$BACKEND"
OUT_DIR=$(python -c "from multi3ir.training.spin.train import cache_path; print(cache_path('$BACKEND', $LAYER))")
LOGD=results/logs/precompute/$(basename "$(dirname "$OUT_DIR")")_layer$LAYER; mkdir -p "$OUT_DIR" "$LOGD"
TARGETS=""; [ "$BACKEND" = qwen ] && TARGETS="--with-targets"
for i in ${SHARD_LIST:-$(seq 0 $((SHARDS - 1)))}; do
  nn=$(printf %02d "$i"); [ -s "$OUT_DIR/shard_$nn.pt" ] && { echo "[skip] shard_$nn"; continue; }
  $RUN python -u -m multi3ir.training.spin.precompute hl --backend "$BACKEND" --layer "$LAYER" --max-length "$MAXLEN" \
      --shard-idx "$i" --num-shards "$SHARDS" --out "$OUT_DIR/shard_$nn.pt" $TARGETS 2>&1 | tee "$LOGD/shard_$nn.log"
done
[ "$(ls "$OUT_DIR"/shard_*.pt 2>/dev/null | wc -l)" -eq "$SHARDS" ] && python -m multi3ir.training.spin.precompute repack --cache-dir "$OUT_DIR" --out "$OUT_DIR"
echo "[precompute] $OUT_DIR"
