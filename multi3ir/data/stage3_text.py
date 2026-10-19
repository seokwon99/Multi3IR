"""Stage 3a: text evidence from C4 — BM25 top-100 per perspective (Pyserini), then dense rerank to top-10 (Qwen3-Embedding-4B).

    # pyserini venv (Java 21)
    python -m multi3ir.data.stage3_text --step bm25-prepare --c4-arrow-dir <hf cache of allenai/c4>   # arrow -> jsonl
    python -m multi3ir.data.stage3_text --step bm25-index
    python -m multi3ir.data.stage3_text --step bm25-search --start_chunk 0 --end_chunk 10               # parallel over chunk ranges
    python -m multi3ir.data.stage3_text --step bm25-finalize                                            # -> c4_bm25_top100_min4.jsonl
    # qwen venv (vLLM)
    python -m multi3ir.data.stage3_text --step dense-prepare                                            # unique docs -> 50k-doc chunks
    python -m multi3ir.data.stage3_text --step dense-encode --chunks 0-9                                # GPU, parallel over ranges
    python -m multi3ir.data.stage3_text --step dense-queries                                            # GPU
    python -m multi3ir.data.stage3_text --step dense-aggregate                                          # -> c4_dense_reranked_top10_min4.jsonl

--bm25 / --chunks-dir / --output override the paths for per-shard runs (scripts/run_shard.sh).
"""
import argparse
import glob
import json
import os
import subprocess
import time
from multiprocessing import Pool

import numpy as np
import tqdm

from multi3ir import paths as _P

C4_JSONL = _P.STAGE3_C4 / "c4_pyserini_jsonl"
C4_INDEX = _P.STAGE3_C4 / "c4_pyserini_index"
BM25_OUT = _P.STAGE3_C4 / "c4_bm25_top100_min4.jsonl"
CHUNKS = _P.STAGE3_C4 / "c4_rerank_chunks_min4"
TOP10_OUT = _P.STAGE3_C4 / "c4_dense_reranked_top10_min4.jsonl"
DENSE_MODEL = "Qwen/Qwen3-Embedding-4B"
QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
MAX_DOC_CHARS_INDEX, MAX_DOC_CHARS_DENSE, PREVIEW_CHARS, CHUNK_SIZE = 10000, 2000, 500, 50000


def load_perspectives(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        for i, p in enumerate(d["perspectives"]):
            out.append({"uid": d["uid"], "perspective_idx": i, "question": d.get("Question", d.get("Title", "")),
                        "summary": p["summary"] if isinstance(p, dict) else p[0]})
    return out


def _read_c4_file(task):
    file_idx, needed, max_len = task
    path = C4_JSONL / f"c4-train-{file_idx}-of-01637.jsonl"
    out = {}
    if path.exists():
        for line in open(path):
            d = json.loads(line)
            if d["id"] in needed:
                out[d["id"]] = (d.get("url", ""), d.get("contents", "")[:max_len])
                if len(out) == len(needed):
                    break
    return out


def read_c4(doc_ids, max_len, workers):
    """doc_id -> (url, text[:max_len]) by streaming only the C4 files those ids live in (c4-<file>-<row>)."""
    by_file = {}
    for did in doc_ids:
        by_file.setdefault(did.split("-")[1], set()).add(did)
    docs = {}
    with Pool(workers) as pool:
        for part in tqdm.tqdm(pool.imap_unordered(_read_c4_file, [(f, s, max_len) for f, s in by_file.items()]),
                              total=len(by_file), desc="c4 files"):
            docs.update(part)
    print(f"  loaded {len(docs):,}/{len(doc_ids):,} docs from {len(by_file):,} files", flush=True)
    return docs


def embed(model, texts):
    e = np.array([o.outputs.embedding for o in model.embed(texts)], dtype=np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


def load_dense_model():
    from vllm import LLM
    return LLM(model=DENSE_MODEL, runner="pooling", dtype="float16", trust_remote_code=True, max_model_len=4096,
               max_num_batched_tokens=4096, enforce_eager=True, gpu_memory_utilization=0.9)


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _arrow_to_jsonl(arrow_path):
    import pyarrow as pa
    stem = os.path.basename(arrow_path)[:-6]                 # c4-train-00042-of-01637
    out = C4_JSONL / f"{stem}.jsonl"
    if out.exists():
        return 0
    table = pa.ipc.open_stream(arrow_path).read_all()
    with open(out, "w") as f:
        for row in range(table.num_rows):
            text = table["text"][row].as_py()
            if text:
                f.write(json.dumps({"id": f"c4-{stem.split('-')[2]}-{row:06d}", "contents": text[:MAX_DOC_CHARS_INDEX],
                                    "url": table["url"][row].as_py() or ""}, ensure_ascii=False) + "\n")
    return table.num_rows


def bm25_prepare(a):
    C4_JSONL.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.c4_arrow_dir, "c4-train-*.arrow")))
    with Pool(a.workers) as pool:
        n = sum(tqdm.tqdm(pool.imap_unordered(_arrow_to_jsonl, files), total=len(files), desc="arrow->jsonl"))
    print(f"[bm25-prepare] {len(files)} files, {n:,} rows -> {C4_JSONL}")


def bm25_index(a):
    subprocess.run(["python", "-m", "pyserini.index.lucene", "--collection", "JsonCollection", "--input", str(C4_JSONL),
                    "--index", str(C4_INDEX), "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(a.workers),
                    "--storeRaw"], check=True)


def bm25_search(a):
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(C4_INDEX))
    searcher.set_bm25(k1=0.9, b=0.4)
    queries = [p["summary"] for p in load_perspectives(_P.PERSPECTIVES)]
    n_chunks = (len(queries) + a.query_chunk_size - 1) // a.query_chunk_size
    end = n_chunks if a.end_chunk is None else min(n_chunks, a.end_chunk)
    chunk_dir = str(BM25_OUT) + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    for ci in range(a.start_chunk, end):
        out = os.path.join(chunk_dir, f"chunk_{ci:03d}.json")
        if os.path.exists(out):
            continue
        s, e = ci * a.query_chunk_size, min((ci + 1) * a.query_chunk_size, len(queries))
        t0 = time.time()
        hits = searcher.batch_search(queries[s:e], [str(i) for i in range(s, e)], k=a.top_k, threads=a.workers)
        json.dump({q: [{"docid": h.docid, "score": h.score} for h in hs] for q, hs in hits.items()}, open(out, "w"))
        print(f"  chunk {ci+1}/{n_chunks} ({time.time()-t0:.0f}s)", flush=True)


def bm25_finalize(a):
    perspectives = load_perspectives(_P.PERSPECTIVES)
    hits = {}
    for cf in sorted(os.listdir(str(BM25_OUT) + ".chunks")):
        hits.update({int(q): h for q, h in json.load(open(os.path.join(str(BM25_OUT) + ".chunks", cf))).items()})
    meta = read_c4({h["docid"] for hs in hits.values() for h in hs}, PREVIEW_CHARS, a.workers)
    with open(BM25_OUT, "w") as f:
        for qid, p in enumerate(perspectives):
            docs = [{"doc_id": h["docid"], "score": h["score"], "rank": r, "url": meta.get(h["docid"], ("", ""))[0],
                     "text_preview": meta.get(h["docid"], ("", ""))[1]} for r, h in enumerate(hits.get(qid, []))]
            f.write(json.dumps({**p, "retrieved_documents": docs}, ensure_ascii=False) + "\n")
    print(f"[bm25-finalize] {len(perspectives):,} rows -> {BM25_OUT}")


# ── dense rerank ──────────────────────────────────────────────────────────────

def dense_prepare(a):
    os.makedirs(a.chunks_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(a.bm25)]
    doc_ids = {d["doc_id"] for r in rows for d in r["retrieved_documents"]}
    print(f"[dense-prepare] {len(rows):,} perspectives, {len(doc_ids):,} unique docs", flush=True)
    qpath = os.path.join(a.chunks_dir, "queries.jsonl")
    if not os.path.exists(qpath):
        with open(qpath, "w") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": f"q_{i}", "text": QUERY_INSTRUCTION + r["summary"]}, ensure_ascii=False) + "\n")
    if glob.glob(os.path.join(a.chunks_dir, "chunk_*.jsonl")):
        print("  doc chunks exist, skipping"); return
    docs = list(read_c4(doc_ids, MAX_DOC_CHARS_DENSE, a.workers).items())
    for ci in range(0, len(docs), CHUNK_SIZE):
        with open(os.path.join(a.chunks_dir, f"chunk_{ci // CHUNK_SIZE:04d}.jsonl"), "w") as f:
            for did, (_, text) in docs[ci:ci + CHUNK_SIZE]:
                f.write(json.dumps({"id": did, "text": text}, ensure_ascii=False) + "\n")
    print(f"  {(len(docs) + CHUNK_SIZE - 1) // CHUNK_SIZE} chunks -> {a.chunks_dir}")


def dense_encode(a):
    lo, _, hi = a.chunks.partition("-")
    todo = [p for ci in range(int(lo), int(hi or lo) + 1)
            if os.path.exists(p := os.path.join(a.chunks_dir, f"chunk_{ci:04d}.jsonl")) and not os.path.exists(p[:-6] + ".npz")]
    if not todo:
        print("[dense-encode] nothing to do"); return
    from transformers import AutoTokenizer
    model, tok = load_dense_model(), AutoTokenizer.from_pretrained(DENSE_MODEL, trust_remote_code=True)
    for path in todo:
        ids, texts = [], []
        for line in open(path):
            d = json.loads(line)
            toks = tok.encode(d["text"])
            ids.append(d["id"]); texts.append(tok.decode(toks[:4000]) if len(toks) > 4000 else d["text"])
        t0 = time.time()
        np.savez(path[:-6] + ".npz", ids=np.array(ids), embeddings=embed(model, texts))
        print(f"  {os.path.basename(path)}: {len(ids):,} docs in {time.time()-t0:.0f}s", flush=True)


def dense_queries(a):
    out = os.path.join(a.chunks_dir, "queries.npz")
    if os.path.exists(out):
        print("[dense-queries] exists"); return
    rows = [json.loads(l) for l in open(os.path.join(a.chunks_dir, "queries.jsonl"))]
    np.savez(out, ids=np.array([r["id"] for r in rows]), embeddings=embed(load_dense_model(), [r["text"] for r in rows]))
    print(f"[dense-queries] {len(rows):,} -> {out}")


def dense_aggregate(a):
    import torch
    npz = sorted(glob.glob(os.path.join(a.chunks_dir, "chunk_*.npz")))
    n_jsonl = len(glob.glob(os.path.join(a.chunks_dir, "chunk_*.jsonl")))
    if len(npz) < n_jsonl:
        raise RuntimeError(f"{n_jsonl - len(npz)} chunks not encoded yet")
    ids, embs = [], []
    for p in tqdm.tqdm(npz, desc="doc embeds"):
        z = np.load(p); ids.extend(z["ids"].tolist()); embs.append(z["embeddings"])
    D = torch.from_numpy(np.concatenate(embs)); idx = {d: i for i, d in enumerate(ids)}
    Q = torch.from_numpy(np.load(os.path.join(a.chunks_dir, "queries.npz"))["embeddings"])
    n_with = 0
    with open(a.output, "w") as f:
        for qi, line in enumerate(tqdm.tqdm(open(a.bm25), desc="rerank")):
            p = json.loads(line)
            cands = [d for d in p["retrieved_documents"] if d["doc_id"] in idx]
            docs = []
            if cands:
                scores = Q[qi:qi + 1] @ D[[idx[d["doc_id"]] for d in cands]].T
                scores = scores.squeeze(0)
                for r, j in enumerate(torch.argsort(scores, descending=True)[:a.top_k]):
                    docs.append({"rank": r, "dense_score": float(scores[j]), "bm25_score": cands[j]["score"],
                                 "doc_id": cands[j]["doc_id"], "url": cands[j].get("url", ""),
                                 "text_preview": cands[j].get("text_preview", "")})
                n_with += 1
            f.write(json.dumps({k: p[k] for k in ("uid", "perspective_idx", "question", "summary")} |
                               {"retrieved_documents": docs}, ensure_ascii=False) + "\n")
    print(f"[dense-aggregate] {qi+1:,} perspectives ({n_with:,} with docs) -> {a.output}")


STEPS = {"bm25-prepare": bm25_prepare, "bm25-index": bm25_index, "bm25-search": bm25_search, "bm25-finalize": bm25_finalize,
         "dense-prepare": dense_prepare, "dense-encode": dense_encode, "dense-queries": dense_queries, "dense-aggregate": dense_aggregate}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=list(STEPS))
    ap.add_argument("--c4-arrow-dir", help="bm25-prepare: directory of c4-train-*.arrow files")
    ap.add_argument("--workers", type=int, default=16, help="CPU workers / index threads / search threads")
    ap.add_argument("--top_k", type=int, default=None, help="bm25-search: 100, dense-aggregate: 10")
    ap.add_argument("--query_chunk_size", type=int, default=10000)
    ap.add_argument("--start_chunk", type=int, default=0); ap.add_argument("--end_chunk", type=int, default=None)
    ap.add_argument("--chunks", help="dense-encode: chunk range, e.g. 0-9")
    ap.add_argument("--bm25", default=str(BM25_OUT)); ap.add_argument("--chunks-dir", default=str(CHUNKS))
    ap.add_argument("--output", default=str(TOP10_OUT))
    a = ap.parse_args()
    if a.top_k is None:
        a.top_k = 100 if a.step == "bm25-search" else 10
    if a.step == "bm25-prepare" and not a.c4_arrow_dir:
        raise SystemExit("bm25-prepare needs --c4-arrow-dir")
    if a.step == "dense-encode" and not a.chunks:
        raise SystemExit("dense-encode needs --chunks (e.g. 0-9)")
    STEPS[a.step](a)


if __name__ == "__main__":
    main()
