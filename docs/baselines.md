# Baselines, judge validation

Run from the checkout with the backbone's venv active and `PYTHONPATH=$PWD`; `BENCH=results/bench`. Every query npz below is scored like SPIN: `python -m multi3ir.eval.run_eval --query-type {naive|spin|oracle} --query-npz <npz> --tag <name> --mmr 0.2` (README, Evaluate).

## Retrieval baselines

**Frozen retriever (naive)** — one vector per question.

```bash
python -m multi3ir.eval.encode --backend qwen --field query --src $BENCH/eval/queries_1039.jsonl --out q.npz      # --query-type naive
```

**Oracle** — the gold perspective descriptions as queries (upper bound).

```bash
python -m multi3ir.eval.encode --backend qwen --field perspectives --src $BENCH/eval/positives_specific.jsonl --out oracle.npz   # --query-type oracle
```

**Naive fine-tuning** — query encoder trained against the frozen document embeddings (`scripts/ft_train.sh`).

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 -m multi3ir.training.baselines.finetune \
    --backend qwen --out results/ckpts/<run> --mode full --lr 1e-5 --tau 0.05 --num-steps 400 --batch-size 16
python -m multi3ir.eval.encode --backend qwen --field query --ckpt results/ckpts/<run>/<ckpt>.pt --src $BENCH/eval/queries_1039.jsonl --out ft_q.npz
```

**ARE** — Auto-Regressive Embedding ([Chen et al. 2025](https://arxiv.org/abs/2511.02770)): a LoRA-tuned decoder emits K query embeddings, trained with Hungarian-matched InfoNCE (`scripts/are_train.sh`).

```bash
python -m multi3ir.training.baselines.are.train --out results/ckpts/<run> --K 5 --backbone qwen --lr 3e-5 --tau 0.01
python -m multi3ir.training.baselines.are.encode --ckpt results/ckpts/<run>/<ckpt>.pt --out are_q.npz --K 5      # --query-type spin
```

**LLM query expansion** — SFT Qwen3-4B to write perspectives, encode them as an (N, m, D) query set.

```bash
python -m multi3ir.training.baselines.llm_expansion build-sft --out-dir <dir>
python -m multi3ir.training.baselines.llm_expansion train --out results/ckpts/<run> --train <dir>/sft_train.jsonl --val <dir>/sft_val.jsonl
python -m multi3ir.training.baselines.llm_expansion generate --ckpt results/ckpts/<run> --out persp.jsonl --n 5
python -m multi3ir.eval.encode --backend qwen --field perspectives --layout grouped --src persp.jsonl --out persp_q.npz   # --query-type spin
```

## Corpus

One document store per backbone, `document_embedding/<backbone>/corpus_docs{,_text}.npz` (images / text; `{ids, embeddings}`), shared by `run_eval` and by the validation inside `spin.train`. It is built from the Hub url tables with `stage4_embed` (README, step 1), or derived from full bench + gold stores that already exist locally:

```bash
python -m multi3ir.eval.corpus_variant qwen        # -> document_embedding/qwen3vl8b/corpus_docs{,_text}.npz
```

## Judge vs. human agreement

The document-support judge (`multi3ir/data/stage4_judge.py`) against 300 human-labeled items (100 balanced for precision, 200 raw-pool for recall): Qwen3-VL-30B-A3B-Instruct reaches agreement 0.90 / precision 0.93 / recall 0.62 on `fully` (paper Table 11); the GPT-5-mini soft-coverage judge reaches κ = 0.84 against the human majority on 100 pairs. Verdicts land in `results/human_annotate/soft_eval/`.

```bash
python -m multi3ir.eval.judge_agreement --backend vllm --model Qwen/Qwen3-VL-30B-A3B-Instruct --tp 2
python -m multi3ir.eval.judge_agreement --backend openai --model gpt-5-mini
python -m multi3ir.eval.judge_agreement --score
```
