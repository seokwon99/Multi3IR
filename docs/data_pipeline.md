# Building Multi³IR-Bench

StackExchange questions → perspectives (gpt-5-mini) → validity + uniqueness filter → text (C4) and image (Serper) retrieval → two-stage support judge → gold. Code: `multi3ir/data/`, one file per step; artifacts: `results/outputs/<stage>/`. Run everything from the repo root with `PYTHONPATH=$PWD` in the qwen venv unless noted.

## 1. StackExchange questions

```bash
python -m multi3ir.data.stage1_stackexchange --base_dir results/outputs/stage1_raw_data/stackexchange_filtered
```

Downloads the archive.org dumps of the 77 sites in `multi3ir/data/stage1_sites.json` (5 categories) and keeps questions with ≥3 answers, ≥1 answer scoring ≥3 and ≥2 links: `<base_dir>/<category>/<site>/<site>_filtered.jsonl`.

## 2. Perspectives

```bash
python -m multi3ir.data.stage2_extract build --src results/outputs/stage1_raw_data/stackexchange_filtered --out results/outputs/stage2_perspective_extraction
# upload <out>/gpt-5-mini-2025-08-07_batch_inputs.jsonl to the OpenAI Batch API, download the results, then
python -m multi3ir.data.stage2_extract parse --src ... --out ... --batch-output <results.jsonl>
```

Each answer set becomes a list of `(summary, [(fact, link)])` perspectives; 2,000 questions sampled per category.

```bash
python -m multi3ir.data.stage2_filter validity --input <shard.jsonl> --output <validity.jsonl>        # MiniCheck-7B vs the answers, one GPU per shard
python -m multi3ir.data.stage2_filter encode   --input <validity.jsonl> --out_dir <emb_dir>          # MiniLM embeddings of the valid summaries
python -m multi3ir.data.stage2_filter final-filter --validity_jsonl <validity.jsonl> --relevance_jsonl <relevance.jsonl> \
    --emb_dir <emb_dir> --output results/outputs/stage2_perspective_extraction/perspective_extracted/final/all_perspectives_filtered_min4.jsonl
```

Keeps perspectives that are valid, relevant and pairwise cosine < 0.8 (greedy keep-first, all-MiniLM-L6-v2); questions with ≥4 survivors remain. `dedup` does encode + greedy dedup in one pass. **Result: 104.9K questions / 521.7K perspectives (72K train / 32K validation / 1.0K test)** — the input of every later step (`paths.PERSPECTIVES`).

## 3. Retrieval

**Text (C4).** BM25 top-100 per perspective, then Qwen3-Embedding-4B rerank to top-10.

```bash
# pyserini venv (Java 21)
python -m multi3ir.data.stage3_text --step bm25-prepare --c4-arrow-dir <hf cache of allenai/c4>
python -m multi3ir.data.stage3_text --step bm25-index
python -m multi3ir.data.stage3_text --step bm25-search --start_chunk 0 --end_chunk 10      # several workers over disjoint chunk ranges
python -m multi3ir.data.stage3_text --step bm25-finalize                                   # -> c4_bm25_top100_min4.jsonl
# qwen venv
python -m multi3ir.data.stage3_text --step dense-prepare                                   # unique docs -> 50k-doc chunks
python -m multi3ir.data.stage3_text --step dense-encode --chunks 0-9                       # one GPU per worker, disjoint ranges
python -m multi3ir.data.stage3_text --step dense-queries
python -m multi3ir.data.stage3_text --step dense-aggregate                                 # -> c4_dense_reranked_top10_min4.jsonl
```

**Images (Serper).** 10 Google-Images results per perspective (key in `$SERPER_API_KEY` or `serper_api.txt`), then a sharded download.

```bash
python -m multi3ir.data.stage3_image search                                                # -> serper_search_min4.jsonl (resumable)
python -m multi3ir.data.stage3_image download --shard 0 --num-shards 8                     # -> serper_images_min4/serper_<hash>.jpg
```

## 4. Support judge → gold

Qwen3-VL-30B-A3B-Instruct labels every (document, perspective) pair `fully` / `partially` / `not`. Stage 1 judges a document against its own perspective; stage 2 re-judges the `fully` documents against every sibling perspective of the same question. **Gold = fully for its own perspective and not fully for any sibling.**

```bash
python -m multi3ir.data.stage4_judge shard --source both --n_shards 10                     # 10 uid-balanced shards of the retrieval files
bash scripts/run_shard.sh 0                                                                # one shard end to end (dense rerank + both judge stages)
```

`run_shard.sh` chains the dense rerank and the two judge commands below per shard (`results/logs/stage4_filter/shard<N>/`):

```bash
python -m multi3ir.data.stage4_judge judge --stage 1 --source image --tp 4 --num_shards 3 --shard 0 --image_input <image_XX.jsonl> --output <s1.jsonl>
python -m multi3ir.data.stage4_judge judge --stage 2 --source image --tp 4 --image_input results/outputs/stage3_retrieval/serper/serper_search_min4.jsonl \
    --stage1_output <s1.jsonl> --output <s2.jsonl>                                          # stage 2 input = the full file (siblings)
python -m multi3ir.data.stage4_gold merge                                                  # stage1 + stage2 -> gold_docs.jsonl
python -m multi3ir.data.stage4_gold positives                                              # -> results/bench/train/positives_gold.jsonl
```

Round 0 gold: 632,558 documents over 234,932 perspectives. `--lenient` appends the leniency instruction used for re-judging.

## 5. Deep rounds (perspectives still without gold)

Ranks 10–29 of the dense rerank (recomputed from the cached embeddings, no new retrieval) and page 2 of Serper images, judged with the same two stages:

```bash
python -m multi3ir.data.stage4_deep window --shard 0 --lo 10 --hi 30 --perspectives <uncovered.json> --out <judge_input.jsonl> --exclude '<stage-1 glob>'
python -m multi3ir.data.stage3_image deep-search --scope nogold --page 2 && python -m multi3ir.data.stage3_image download --inputs <deep search jsonl> --img-dir results/outputs/stage3_retrieval/serper/images_deep
python -m multi3ir.data.stage4_judge judge --stage 1 --lenient ...                         # as above
python -m multi3ir.data.stage4_deep stage2-input --stage1 '<s1 glob>' --search <judge inputs> --source image --out <s2_input.jsonl>
python -m multi3ir.data.stage4_judge judge --stage 2 --lenient --image_input <s2_input.jsonl> --stage1_output '<s1 glob>' ...
python -m multi3ir.data.stage4_gold round --stage1 '<s1 glob>' --stage2 '<s2 glob>' --stage2-input <s2_input.jsonl> --modality image --round <name> --out results/outputs/stage4_retrieval_filter/deep_nogold/gold_image.jsonl
python -m multi3ir.data.stage4_deep texts --gold gold_text.jsonl gold_text2.jsonl           # body text of the new gold docs
python -m multi3ir.data.stage4_embed queue --model qwen3vl8b && python -m multi3ir.data.stage4_embed encode --model qwen3vl8b   # then `merge`
python -m multi3ir.data.stage4_gold positives-all                                          # every round -> positives_gold_all.jsonl
```

## 6. Subject domains (dataset statistics)

```bash
python -m multi3ir.data.domain_collect text && python -m multi3ir.data.domain_classify topic text          # C4 body -> WebOrganizer topic
python -m multi3ir.data.domain_collect links && python -m multi3ir.data.domain_collect crawl --concurrency 64
python -m multi3ir.data.domain_classify topic image                                                       # hosting-page body -> topic
python -m multi3ir.data.domain_classify title-url                                                          # pages that never loaded: url + title
python -m multi3ir.data.domain_collect gold-input                                                          # all rounds -> gold_all_input.jsonl + gold_key_map.jsonl
python -m multi3ir.data.domain_classify topic input --input results/outputs/domain_classification/gold_all_input.jsonl --model nvidia_ml
python -m multi3ir.data.domain_classify fdc --input results/outputs/domain_classification/gold_all_input.jsonl --out <fdc.jsonl>
```

## 7. Corpus for the release

```bash
python -m multi3ir.data.build_corpus --check
python -m multi3ir.data.build_corpus --images --workers 64                                 # doc_url -> results/bench/corpus/images/<doc_id>
python -m multi3ir.data.build_corpus --text --c4-dir results/outputs/stage3_retrieval/c4/c4_pyserini_jsonl   # -> corpus/text.jsonl
```

## Outputs

```
results/outputs/
├── stage1_raw_data/stackexchange_filtered/<category>/<site>/<site>_filtered.jsonl
├── stage2_perspective_extraction/perspective_extracted/final/all_perspectives_filtered_min4.jsonl
├── stage3_retrieval/c4/       c4_pyserini_{jsonl,index}/ · c4_bm25_top100_min4.jsonl · c4_rerank_chunks_min4_shardXX/ · c4_dense_reranked_top10_min4_shardXX.jsonl
├── stage3_retrieval/serper/   serper_search_min4.jsonl · serper_images_min4/ · images_deep/ · images_textonly/
├── stage4_retrieval_filter/   shards/{image,text}_XX.jsonl · gold_docs.jsonl · deep_nogold/gold_*.jsonl
└── domain_classification/     full_topics_*.jsonl · gold_all_input.jsonl · gold_key_map.jsonl
results/bench/train/           queries.jsonl · positives_gold.jsonl · positives_gold_all.jsonl · heldout_uids.txt
```
