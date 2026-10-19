<h2 align="center">[EMNLP '26] Multi³IR: A Benchmark for Multi-perspective<br>Multi-domain Multi-modal Information Retrieval</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2608.30949"><img src="https://img.shields.io/badge/arXiv-2608.30949-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/abs/2608.30949"><img src="https://img.shields.io/badge/EMNLP-2026-blue.svg" alt="EMNLP 2026"></a>
  <a href="https://seokwon99.github.io/Multi3IR/"><img src="https://img.shields.io/badge/Project-Page-3b5bdb.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/seokwon99/Multi3IR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Multi%C2%B3IR--Bench-FFD21E" alt="Multi³IR-Bench on Hugging Face"></a>
</p>

<p align="center">
  <b>Open-ended queries have many valid perspectives — retrieve all of them, across domains and modalities.</b>
</p>

<h5 align="center">If you find this project useful, please consider giving it a star ⭐.</h5>

## 💡 Overview

Information retrieval increasingly targets open-ended queries that admit diverse perspectives. Existing IR benchmarks focus on closed-ended queries, and even open-ended ones mostly have supporting documents from a single subject domain and a single modality.

**Multi³IR** is a benchmark that evaluates how well retrievers cover the multifaceted perspectives of open-ended queries across diverse domains and modalities. It comprises 104.9K Stack Exchange queries (72K train / 32K validation / 1.0K test), each annotated with perspective descriptions that capture the query's implicit viewpoints (521.7K perspectives, 4.98 per query), over a 1.01M-document corpus of C4 web text (631.4K) and web images (380.7K) with judged gold documents per perspective.

**SPIN** (Steering Perspectives by Injecting Noise) is a parameter- and label-efficient method that learns noise vectors to steer a frozen retriever's embeddings toward diverse yet meaningful semantic directions. Existing multimodal retrievers suffer from single-perspective bias; SPIN substantially improves perspective coverage on Multi³IR and generalizes to unseen open-ended IR benchmarks.

<p align="center"><img src="assets/multi3ir.gif" width="100%" alt="An open-ended query branches into perspectives; each perspective is searched and supported by documents from different domains and modalities."></p>

## ✨ Highlights

- **Multi-perspective:** 104.9K questions (72K / 32K / 1.0K train / validation / test), 4.98 perspective descriptions per question, gold documents judged per perspective (exclusive support).
- **Multi-domain, multi-modal:** 77 Stack Exchange sites in 5 categories; 3.34 subject domains and 1.91 modalities per query on average (C4 text + web images).
- **Coverage metrics:** Hard Coverage HC@k (k = 5, 10, 20, 100; a perspective is covered when one of its gold documents is retrieved) and Soft Coverage SC@k (k = 5, 10, 15, 20; GPT-5-mini judges support, κ = 0.84 against human majority).
- **SPIN:** *m* = 5 learned noise vectors ε ∈ ℝ^(m×H) injected at layer 0.5L of a frozen encoder — 20.5K trainable parameters, no retriever fine-tuning (Qwen3-VL-Embedding-8B, GME-Qwen2-VL-7B, MM-Embed).
- **Full pipeline:** benchmark construction, SPIN training and baselines (Naive, Naive fine-tuned, LLM-Expansion, ARE, Oracle) are all reproducible from this repository.

## 🔧 Installation

The reference environment uses **Python 3.12**, **PyTorch 2.8** and **CUDA 12.8**.

```bash
git clone https://github.com/seokwon99/Multi3IR.git && cd Multi3IR

python -m venv envs/qwen3vl_embed && source envs/qwen3vl_embed/bin/activate
pip install -r requirements/qwen.txt
pip install flash-attn==2.8.3 --no-build-isolation
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git envs/qwen3vl_embedding_repo   # imported at runtime
pip install -e .

python -m multi3ir.paths        # prints every expected location
```

GME and MM-Embed need their own venvs (`requirements/gme.txt`, `requirements/mmembed.txt`, conflicting `transformers` pins) at `envs/gme_reference/.venv` and `envs/mmembed_env/.venv`. `results/` and `envs/` may be symlinks to another disk; `MULTI3IR_ROOT` relocates both.

## 🏃 Quick Start

### 1. Get the benchmark

The benchmark lives on the Hub as [seokwon99/Multi3IR](https://huggingface.co/datasets/seokwon99/Multi3IR): queries, perspectives and gold labels as parquet tables, documents as (doc_id, url) — text bodies and image bytes are not redistributed.

```bash
python -m multi3ir.data.hub pull                                   # -> results/bench/{train,eval,corpus}
python -m multi3ir.data.build_corpus --images --workers 64         # image documents from their urls
python -m multi3ir.data.build_corpus --text --c4-dir <c4_jsonl>    # C4 passages by doc_id (eval gold passages are inline)
python -m multi3ir.data.stage4_embed queue --model qwen3vl8b       # document embeddings: queue → encode (one GPU per worker) → merge
python -m multi3ir.data.stage4_embed encode --model qwen3vl8b
python -m multi3ir.data.stage4_embed merge --model qwen3vl8b       # -> document_embedding/qwen3vl8b/corpus_docs{,_text}.npz
```

```text
results/bench/
├── train/queries.jsonl · train/positives_gold.jsonl      questions, perspectives, gold documents
├── eval/queries_1039.jsonl · eval/positives_specific.jsonl
├── corpus/text.jsonl · corpus/images/                     built above
└── document_embedding/qwen3vl8b/                          encoded above
```

### 2. Train SPIN

```bash
CACHE=results/bench/precompute/qwen3vl8b/layer18      # hidden-state cache of the training queries at the injection layer

# cache hidden states at the injection layer (one GPU per shard) and repack
for i in $(seq 0 15); do
  python -m multi3ir.training.spin.precompute hl --backend qwen --layer 18 --with-targets \
      --shard-idx $i --num-shards 16 --out $CACHE/shard_$(printf %02d $i).pt
done
python -m multi3ir.training.spin.precompute repack --cache-dir $CACHE --out $CACHE

# mine hard negatives
python -m multi3ir.training.spin.precompute queries --encoder qwen --out results/ckpts/spin_mining/qwen_full_queries.npz
python -m multi3ir.training.spin.mine --encoder qwen --anchor question --k 2048 \
    --queries results/ckpts/spin_mining/qwen_full_queries.npz --out results/ckpts/spin_mining/qwen_full_question_k2048.npz

# train  → results/ckpts/spin/<run>/spin_best.pt
python -m multi3ir.training.spin.train --encoder qwen --layer 18 --m 5 --data full
```

Defaults follow the paper: `--m 5`, `--layer` = 0.5L (18 qwen / 14 gme / 16 mmembed), ε ~ 𝒩(0, 0.01²) (`--noise-std 0.01`, `--eps-rank 0`), noisy-OR loss with in-batch negatives (`--neg-anchor none`), batch 32, lr 1e-2 / 1e-3 / 3e-4. Run names come from the config (`qwen_s0` = the defaults).

### 3. Evaluate

```bash
# encode the evaluation queries → (N, m, D)
python -m multi3ir.training.spin.encode --encoder qwen --ckpt results/ckpts/spin/<run>/spin_best.pt --out spin_<run>.npz

# HC@k (k = 5, 10, 20, 100) on the test split; slot rankings merged by round robin
python -m multi3ir.eval.run_eval --query-type spin --tag <run> --query-npz spin_<run>.npz --save-retrieval retrieval_<run>.jsonl

# SC@k (k = 5, 10, 15, 20) with the GPT-5-mini judge — needs OPENAI_API_KEY
python -m multi3ir.eval.soft_coverage prepare --retrievals retrieval_<run>.jsonl --topk 20
python -m multi3ir.eval.soft_coverage judge --model gpt-5-mini --workers 32
python -m multi3ir.eval.soft_coverage score --retrieval retrieval_<run>.jsonl --ks 5 10 15 20
```

A frozen retriever: `python -m multi3ir.eval.encode --backend qwen --field query --src results/bench/eval/queries_1039.jsonl --out q.npz`, then score with `--query-type naive`. `scripts/spin_precompute.sh`, `spin_mine.sh`, `spin_train.sh`, `spin_eval.sh` wrap the three steps; `RUN="srun --gres=gpu:1"` (any job-submission prefix) sends each GPU command to a scheduler.

## 🧩 How SPIN Works

A frozen encoder is split at layer *l* = 0.5L. For a query, the hidden states at *l* are perturbed *m* times with learned noise vectors ε₁…ε_m ∈ ℝ^H (initialised from 𝒩(0, 0.01²)) and the remaining layers are run once per slot, giving *m* unit embeddings per query instead of one.

- **ε** — one global (m, H) table shared by every query (`--eps-rank 0`, 20.5K parameters for m = 5). A per-query bottleneck variant is available with `--eps-rank r`.
- **Set-coverage loss** — noisy-OR: ℒ_pos asks every target perspective embedding to be covered by *any* slot, `-log(1 − ∏ₛ(1 − pₛ))`; ℒ_neg asks every slot to reject the in-batch negatives.
- **Retrieval** — the *m* slot rankings are merged by round robin (rank-1 of every slot, then rank-2, …), so the final list covers several perspectives instead of repeating the dominant one.

Only ε (or the bottleneck) is trained; the encoder, the document index and the tokenizer stay untouched, so SPIN drops into any dense retriever.

## 📚 About Benchmark

**Multi³IR-Bench** — the 1.0K-question test split (4.8K perspectives, 12.5K supporting documents) over the 1.01M-document corpus. Baselines (Naive, Naive fine-tuned, LLM-Expansion, ARE, Oracle): [docs/baselines.md](docs/baselines.md).

It is built by a five-stage pipeline (`multi3ir/data/`, one file per step): Stack Exchange questions → perspective extraction (GPT-5-mini) → faithfulness (Bespoke-MiniCheck-7B) + uniqueness (all-MiniLM-L6-v2, cosine > 0.8 removed) filters → text (C4, BM25 + dense rerank) and image (Google Image Search via Serper) retrieval → two-stage support judge (Qwen3-VL-30B-A3B-Instruct: supports its own perspective and no sibling) → gold documents. Step-by-step commands: [docs/data_pipeline.md](docs/data_pipeline.md).

## 📁 Repository Structure

```text
Multi3IR/
├── multi3ir/
│   ├── paths.py             # every data / checkpoint / log location
│   ├── data/                # benchmark construction: stage1_stackexchange → stage2_* → stage3_* → stage4_* → build_corpus
│   ├── training/
│   │   ├── spin/            # precompute → mine → train → encode; model.py, loss.py, data.py
│   │   ├── backbones/       # frozen encoders split at the injection layer: qwen, mmembed, gme
│   │   └── baselines/       # finetune, are/, llm_expansion
│   └── eval/                # encode, run_eval, soft_coverage, corpus_variant, hard_coverage, judge_agreement, berds/, pir/
├── scripts/                 # launchers, one per step; RUN="srun --gres=gpu:1" submits them to a scheduler
├── docs/                    # data_pipeline.md, baselines.md
├── requirements/            # qwen.txt, gme.txt, mmembed.txt
├── results/                 # bench/, outputs/, ckpts/, logs/, other_bench/, human_annotate/   (not tracked)
└── envs/                    # one virtualenv per backbone                                        (not tracked)
```

## 📖 Citation

If you find Multi³IR or SPIN useful, please cite our paper:

```bibtex
@article{song2026multi3ir,
  title={MULTI3IR: A Benchmark for Multi-perspective Multi-domain Multi-modal Information Retrieval},
  author={Song, Seokwon and Kim, Sohyeon and Kim, Gunhee},
  journal={arXiv preprint arXiv:2608.30949},
  year={2026}
}
```

## ❤️ Acknowledgements

This project builds on [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding), [GME](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) and [MM-Embed](https://huggingface.co/nvidia/MM-Embed) as retrievers, on Stack Exchange, C4 and the Serper API as data sources, and on [MiniCheck](https://github.com/Liyan06/MiniCheck) and Qwen3-VL for perspective and document judging.

## 📞 Contact

For questions or feedback, please open a GitHub issue or contact the first author (see the paper).
