#!/bin/bash
# One stage-4 shard end to end: text = dense rerank -> 2-stage judge; image = 2-stage judge.
#   bash scripts/run_shard.sh 0 [image|text|both]      # needs stage4_retrieval_filter/shards/{image,text}_XX.jsonl (stage4_judge shard)
# TP = GPUs per judge process (4). Logs and verdicts: results/logs/stage4_filter/shard<N>/
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH=$PWD
RUN=${RUN:-}                       # job-submission prefix for every GPU command, e.g. RUN="srun --gres=gpu:1"
venv() { source "$(dirname "$(dirname "$(python3 -c "from multi3ir.paths import env_python; print(env_python(\"$1\"))")")")/bin/activate"; }
SHARD=${1:?usage: run_shard.sh <shard> [image|text|both]}; WHICH=${2:-both}; XX=$(printf %02d "$SHARD"); TP=${TP:-4}
venv qwen
OUT=results/outputs; C4=$OUT/stage3_retrieval/c4; SHARDS=$OUT/stage4_retrieval_filter/shards
LOGS=results/logs/stage4_filter/shard$SHARD; mkdir -p "$LOGS"
CHUNKS=$C4/c4_rerank_chunks_min4_shard$XX; TOP10=$C4/c4_dense_reranked_top10_min4_shard$XX.jsonl
JUDGE="python -u -m multi3ir.data.stage4_judge judge --tp $TP --batch_size 3000"
DENSE="python -u -m multi3ir.data.stage3_text --bm25 $SHARDS/text_$XX.jsonl --chunks-dir $CHUNKS --output $TOP10"
judge_two_stages() {   # <source> <stage-1 input> <stage-2 input = the full retrieval file, for sibling lookup>
  $RUN $JUDGE --stage 1 --source "$1" "$2" --output "$LOGS/$1_stage1.jsonl" 2>&1 | tee "$LOGS/$1_stage1.log"
  $RUN $JUDGE --stage 2 --source "$1" "$3" --stage1_output "$LOGS/$1_stage1.jsonl" --output "$LOGS/$1_stage2.jsonl" 2>&1 | tee "$LOGS/$1_stage2.log"
}
run_image() { judge_two_stages image "--image_input $SHARDS/image_$XX.jsonl" "--image_input $OUT/stage3_retrieval/serper/serper_search_min4.jsonl"; }
run_text() {
  $DENSE --step dense-prepare 2>&1 | tee "$LOGS/text_prepare.log"
  $RUN $DENSE --step dense-encode 2>&1 | tee "$LOGS/text_encode.log"            # --chunks A-B to split over workers
  $RUN $DENSE --step dense-queries 2>&1 | tee "$LOGS/text_queries.log"
  $DENSE --step dense-aggregate 2>&1 | tee "$LOGS/text_aggregate.log"
  judge_two_stages text "--text_input $TOP10" "--text_input $C4/c4_bm25_top100_min4.jsonl"
}
case $WHICH in image) run_image ;; text) run_text ;; both) run_image; run_text ;; *) echo "image|text|both" >&2; exit 2 ;; esac
echo "[shard $SHARD] complete"
