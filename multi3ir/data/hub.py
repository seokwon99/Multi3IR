"""Multi³IR-Bench <-> the Hugging Face Hub (seokwon99/Multi3IR).

    python -m multi3ir.data.hub pull                                   # Hub -> results/bench/{train,eval,corpus}
    python -m multi3ir.data.hub push --out results/hf_release          # results/bench -> parquet configs + card (+ --upload)

Documents travel as (doc_id, url) only — no text bodies or image bytes, except the inline text of the eval gold
docs. After `pull`, `python -m multi3ir.data.build_corpus` fetches the corpus itself.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multi3ir import paths as _P

REPO = "seokwon99/Multi3IR"
SERPER_FILES = ("serper_search_min4.jsonl", "image_deep_nogold.jsonl",
                "image_textonly_page1.jsonl", "image_textonly_page2.jsonl")
CONFIGS = {"queries": ("train", "eval"), "gold": ("train", "eval"), "corpus": ("text", "image")}


def jsonl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)


def write(rows, path: Path, schema: pa.Schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    print(f"[push] {path.relative_to(path.parents[1])}  {len(rows):,} rows", flush=True)


# ── push: results/bench -> parquet ────────────────────────────────────────────
def push_queries(out: Path):
    s = pa.schema([("uid", pa.string()), ("query", pa.string()), ("perspectives", pa.list_(pa.string()))])
    for split, src in (("train", _P.TRAIN_QUERIES), ("eval", _P.EVAL_QUERIES)):
        write([{"uid": r["uid"], "query": r["query"], "perspectives": r["positives"]} for r in jsonl(src)],
              out / "queries" / f"{split}.parquet", s)


GOLD = pa.schema([("uid", pa.string()), ("perspective_idx", pa.int32()), ("summary", pa.string()),
                  ("doc_id", pa.string()), ("modality", pa.string()), ("url", pa.string()),
                  ("domain", pa.string()), ("rank", pa.int32()), ("text", pa.string())])


def push_gold(out: Path):
    """One schema for both splits (datasets casts every split of a config to the same features)."""
    summary = {(r["uid"], i): t for r in jsonl(_P.TRAIN_QUERIES) for i, t in enumerate(r["positives"])}
    write([{"uid": r["uid"], "perspective_idx": int(r["perspective_idx"]), "summary": summary[(r["uid"], int(r["perspective_idx"]))],
            "doc_id": r["doc_id"], "modality": r["modality"], "url": r["doc_url"], "domain": r["domain"], "rank": None, "text": None}
           for r in jsonl(_P.TRAIN_POSITIVES)], out / "gold" / "train.parquet", GOLD)
    rows = []
    for r in jsonl(_P.EVAL_GOLD):
        base = {"uid": r["uid"], "perspective_idx": int(r["perspective_idx"]), "summary": r["summary"], "domain": None}
        for d in r["retrieved_documents"]:
            rows.append({**base, "doc_id": d["doc_id"], "modality": "text", "url": d["url"], "rank": int(d["rank"]), "text": d["text"]})
        for d in r["retrieved_images"]:
            rows.append({**base, "doc_id": d["image_name"], "modality": "image", "url": d["image_url"], "rank": int(d["rank"]), "text": None})
    write(rows, out / "gold" / "eval.parquet", GOLD)


def push_corpus(out: Path):
    """The benchmark corpus (corpus_docs{,_text}.npz ids): every doc id with its source url."""
    s = pa.schema([("doc_id", pa.string()), ("url", pa.string())])
    emb = _P.DOC_EMB["qwen"]
    ids = lambda f: {str(x) for x in np.load(emb / f, allow_pickle=True)["ids"]}
    txt, img = ids("corpus_docs_text.npz"), ids("corpus_docs.npz")
    url = {r["doc_id"]: r["url"] for r in jsonl(_P.BENCH / "corpus/text.jsonl") if r["doc_id"] in txt}
    for f in (_P.TRAIN_POSITIVES, _P.BENCH / "train/positives_gold_all.jsonl"):
        for r in jsonl(f):
            if r["modality"] == "text" and r["doc_id"] in txt and r.get("doc_url"):
                url.setdefault(r["doc_id"], r["doc_url"])
    if txt - url.keys():                       # deep-round gold beyond the released text corpus: the BM25 candidate file has every url
        for r in jsonl(_P.STAGE3_C4 / "c4_bm25_top100_min4.jsonl"):
            for d in r["retrieved_documents"]:
                if d["doc_id"] in txt:
                    url.setdefault(d["doc_id"], d["url"])
    missing = txt - url.keys()
    if missing:
        raise RuntimeError(f"{len(missing):,} corpus texts without a url, e.g. {sorted(missing)[:3]}")
    write([{"doc_id": k, "url": url[k]} for k in sorted(txt)], out / "corpus" / "text.parquet", s)
    url = {}
    for f in [_P.STAGE3_SERPER / f for f in SERPER_FILES] + [_P.EVAL_GOLD]:
        for r in jsonl(f):
            for d in r["retrieved_images"]:
                if d["image_name"] in img:
                    url.setdefault(d["image_name"], d["image_url"])
    for r in jsonl(_P.TRAIN_POSITIVES):
        if r["modality"] == "image" and r["doc_id"] in img:
            url.setdefault(r["doc_id"], r["doc_url"])
    missing = img - url.keys()
    if missing:
        raise RuntimeError(f"{len(missing):,} corpus images without a url, e.g. {sorted(missing)[:3]}")
    write([{"doc_id": k, "url": url[k]} for k in sorted(img)], out / "corpus" / "image.parquet", s)


def push_card(out: Path):
    n = lambda c, sp: f"{pq.read_metadata(out / c / f'{sp}.parquet').num_rows:,}"
    cfg = "\n".join(f"  - config_name: {c}\n    data_files:\n" + "".join(
        f"      - split: {s}\n        path: {c}/{s}.parquet\n" for s in splits) for c, splits in CONFIGS.items())
    (out / "README.md").write_text(f"""---
license: cc-by-sa-4.0
language: [en]
task_categories: [text-retrieval]
pretty_name: Multi³IR-Bench
configs:
{cfg}---

# Multi³IR-Bench

Multi-perspective, multi-domain, multi-modal information retrieval benchmark ([paper](https://arxiv.org/abs/2608.30949), [code](https://github.com/seokwon99/Multi3IR)). 104.9K Stack Exchange questions (72K train / 32K validation / 1.0K test) with 521.7K perspective descriptions (4.98 per query) and judged gold documents per perspective, over a 1.01M-document corpus of C4 web text and web images. This release ships the train split (train + validation questions together) and the 1.0K test split.

| config | split | rows | columns |
|---|---|---|---|
| `queries` | train / eval | {n('queries', 'train')} / {n('queries', 'eval')} | uid, query, perspectives |
| `gold` | train / eval | {n('gold', 'train')} / {n('gold', 'eval')} | uid, perspective_idx, summary, doc_id, modality, url, domain (train), rank + text (eval; text inline for text docs) |
| `corpus` | text / image | {n('corpus', 'text')} / {n('corpus', 'image')} | doc_id, url — the benchmark corpus (every document that is gold for some question) |

Documents are referenced by id and source url, not redistributed: C4 passages (`c4-<file>-<line>`) can be recovered from [allenai/c4](https://huggingface.co/datasets/allenai/c4), images from their url. In the code repository, `python -m multi3ir.data.hub pull` restores the benchmark layout and `python -m multi3ir.data.build_corpus` materialises the corpus.

```python
from datasets import load_dataset
gold = load_dataset("seokwon99/Multi3IR", "gold", split="eval")
```

Sources: Stack Exchange (CC BY-SA 4.0), C4 (ODC-BY), images © their owners (research use). Perspectives were written by GPT-5-mini and filtered for faithfulness (Bespoke-MiniCheck-7B) and uniqueness (all-MiniLM-L6-v2, cosine > 0.8); document support was judged by Qwen3-VL-30B-A3B-Instruct (agreement 0.90 with human labels on `fully`).

```bibtex
@article{{song2026multi3ir,
  title={{MULTI3IR: A Benchmark for Multi-perspective Multi-domain Multi-modal Information Retrieval}},
  author={{Song, Seokwon and Kim, Sohyeon and Kim, Gunhee}},
  journal={{arXiv preprint arXiv:2608.30949}},
  year={{2026}}
}}
```
""")


# ── pull: parquet -> results/bench ────────────────────────────────────────────
def pull(repo: str, bench: Path):
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(repo, repo_type="dataset"))
    table = lambda c, s: pq.read_table(src / c / f"{s}.parquet").to_pylist()

    def dump(rows, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[pull] {path.relative_to(bench)}  {len(rows):,} rows", flush=True)

    for split, path in (("train", _P.TRAIN_QUERIES), ("eval", _P.EVAL_QUERIES)):
        dump([{"uid": r["uid"], "query": r["query"], "positives": r["perspectives"]} for r in table("queries", split)],
             bench / path.relative_to(_P.BENCH))
    dump([{"uid": r["uid"], "perspective_idx": r["perspective_idx"], "doc_id": r["doc_id"], "modality": r["modality"],
           "doc_url": r["url"], "domain": r["domain"]} for r in table("gold", "train")],
         bench / _P.TRAIN_POSITIVES.relative_to(_P.BENCH))
    persp = {}
    for r in table("gold", "eval"):
        p = persp.setdefault((r["uid"], r["perspective_idx"]),
                             {"uid": r["uid"], "perspective_idx": r["perspective_idx"], "summary": r["summary"],
                              "retrieved_documents": [], "retrieved_images": []})
        if r["modality"] == "text":
            p["retrieved_documents"].append({"rank": r["rank"], "doc_id": r["doc_id"], "url": r["url"], "text": r["text"]})
        else:
            p["retrieved_images"].append({"rank": r["rank"], "image_name": r["doc_id"], "image_url": r["url"]})
    dump(list(persp.values()), bench / _P.EVAL_GOLD.relative_to(_P.BENCH))
    (bench / "corpus").mkdir(parents=True, exist_ok=True)
    for s in CONFIGS["corpus"]:
        shutil.copy(src / "corpus" / f"{s}.parquet", bench / "corpus" / f"{s}_urls.parquet")
    print(f"[pull] corpus/{{text,image}}_urls.parquet -> next: python -m multi3ir.data.build_corpus --images --text")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pull"); s.add_argument("--repo", default=REPO); s.add_argument("--out", default=str(_P.BENCH))
    s = sub.add_parser("push"); s.add_argument("--out", required=True); s.add_argument("--upload", metavar="REPO")
    a = p.parse_args()
    if a.cmd == "pull":
        pull(a.repo, Path(a.out))
        return
    out = Path(a.out)
    push_queries(out); push_gold(out); push_corpus(out); push_card(out)
    if a.upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=str(out), repo_id=a.upload, repo_type="dataset",
                              commit_message="Multi³IR-Bench release")
        print(f"[push] uploaded -> https://huggingface.co/datasets/{a.upload}")


if __name__ == "__main__":
    main()
