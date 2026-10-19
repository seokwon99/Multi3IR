"""Hard perspective coverage@k on the external benches that ship gold documents (no judge): a perspective is
covered when one of its OWN gold docs is in the question's top-k. Query = original question only.

  python -m multi3ir.eval.hard_coverage --prep                                   # build pir_*/ + arguana/ from raw
  python -m multi3ir.eval.hard_coverage --backend qwen --data pir_allsides
  python -m multi3ir.eval.hard_coverage --backend qwen --data arguana --query-npz spin_q.npz --tag spin
"""
import argparse
import collections
import csv
import hashlib
import json
import os

import numpy as np
import torch

from multi3ir import paths as _P
from multi3ir.eval.encode import BACKENDS, INSTRUCTION, encode
from multi3ir.eval.run_eval import merge_round_robin

DATA = _P.OTHER_BENCH / "hard_coverage"
KS = (5, 10, 20, 100)


# ── prep: queries.jsonl {uid, query} · corpus.jsonl {doc_id, text} · qrels.json uid -> [{perspective, gold}] ──
def write(name, queries, corpus, qrels):
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "queries.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries))
    (d / "corpus.jsonl").write_text("".join(json.dumps({"doc_id": k, "text": t}, ensure_ascii=False) + "\n"
                                            for k, t in corpus.items()))
    (d / "qrels.json").write_text(json.dumps(qrels))
    print(f"[{name}] questions {len(queries):,} | perspectives {sum(map(len, qrels.values())):,} | corpus {len(corpus):,}")


def prep_pir():
    """qrels.tsv gives each perspective query's gold; meta.src_query groups perspectives under their question."""
    raw = _P.OTHER_BENCH / "pir/raw"
    for sub in sorted(os.listdir(raw)):
        corpus = {r["_id"]: (r.get("title", "") + "\n" + r["text"]).strip()
                  for r in map(json.loads, open(raw / sub / "corpus.jsonl"))}
        qmeta = {}
        for r in map(json.loads, open(raw / sub / "queries.jsonl")):
            m = json.loads(r["meta"]) if isinstance(r.get("meta"), str) else (r.get("meta") or {})
            qmeta[r["_id"]] = (m.get("src_query") or r["text"], r["text"])
        gold = collections.defaultdict(list)
        rows = list(csv.reader(open(raw / sub / "qrels.tsv"), delimiter="\t"))[1:]
        for qid, did, sc in rows:
            if float(sc) > 0 and qid in qmeta and did in corpus:
                gold[qid].append(did)
        by_q = collections.defaultdict(dict)
        for qid, (src, persp) in qmeta.items():
            if gold.get(qid):
                by_q[src][persp] = sorted(set(gold[qid]))
        queries, qrels = [], {}
        for i, (src, persps) in enumerate(sorted(by_q.items())):
            queries.append({"uid": f"{sub}-{i}", "query": src})
            qrels[f"{sub}-{i}"] = [{"perspective": p, "gold": g} for p, g in sorted(persps.items())]
        write(f"pir_{sub}", queries, corpus, qrels)


def prep_arguana():
    """BeRDS's only subset with inline gold docs (perspectives[i] pairs with docs[i]); pool = all gold, dedup by text."""
    corpus, queries, qrels = {}, [], {}
    for split in ("dev", "test"):
        for i, r in enumerate(map(json.loads, open(_P.OTHER_BENCH / f"berds/raw/arguana.{split}.jsonl"))):
            persps, docs = r.get("perspectives") or [], r.get("docs") or []
            if not docs or len(docs) != len(persps):
                continue
            uid, entries = f"arguana-{split}-{i}", []
            for p, t in zip(persps, docs):
                did = "ag-" + hashlib.md5(t.encode()).hexdigest()[:12]
                corpus[did] = t
                entries.append({"perspective": p, "gold": [did]})
            queries.append({"uid": uid, "query": r["question"], "split": split})
            qrels[uid] = entries
    write("arguana", queries, corpus, qrels)


# ── score ──────────────────────────────────────────────────────────────────
def embed(backend, texts, instruction, cache, batch):
    if cache.exists():
        return np.load(cache)["e"]
    e = encode(backend, texts, instruction, batch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, e=e)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true", help="build the data folders from other_bench/{pir,berds}/raw")
    ap.add_argument("--backend", choices=sorted(BACKENDS))
    ap.add_argument("--data", help="folder under other_bench/hard_coverage")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default=None, help="result/cache name (default: backend)")
    ap.add_argument("--query-instruction", default=INSTRUCTION)
    ap.add_argument("--query-npz", default=None, help="multi-slot queries (ids, embeddings[N,m,D]); round-robin merged")
    ap.add_argument("--distractors", default=None,
                    help="encoder dir under other_bench/berds/encodings/ whose wikipedia embeddings are appended to the pool")
    a = ap.parse_args()
    if a.prep:
        prep_pir(); prep_arguana()
        return
    if not (a.backend and a.data):
        ap.error("--backend and --data")
    tag, d = a.tag or a.backend, DATA / a.data
    queries = [json.loads(l) for l in open(d / "queries.jsonl")]
    corpus = [json.loads(l) for l in open(d / "corpus.jsonl")]
    qrels = json.load(open(d / "qrels.json"))
    did2row = {c["doc_id"]: i for i, c in enumerate(corpus)}

    slots = None
    if a.query_npz:
        z = np.load(a.query_npz, allow_pickle=True)
        by = {str(u): e for u, e in zip(z["ids"], z["embeddings"])}
        missing = [q["uid"] for q in queries if q["uid"] not in by]
        if missing:
            raise SystemExit(f"{len(missing)} queries missing from {a.query_npz}, e.g. {missing[:3]}")
        slots = np.stack([by[q["uid"]] for q in queries]).astype(np.float32)
        slots /= np.linalg.norm(slots, axis=-1, keepdims=True) + 1e-12
    else:
        qe = embed(a.backend, [q["query"] for q in queries], a.query_instruction, d / f"emb/{tag}_q.npz", a.batch)
    de = embed(a.backend, [c["text"] for c in corpus], "", d / f"emb/{a.backend}_d.npz", a.batch)
    n_gold = de.shape[0]
    if a.distractors:
        dd = np.array(np.load(_P.OTHER_BENCH / f"berds/encodings/{a.distractors}/doc_embs.npy", mmap_mode="r"), dtype=np.float32)
        dd /= np.linalg.norm(dd, axis=1, keepdims=True) + 1e-12
        de = np.concatenate([de, dd])
        print(f"  pool {n_gold:,} gold + {dd.shape[0]:,} distractors", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = torch.from_numpy(de).to(dev, torch.float16).T.contiguous()
    Kmax = min(max(KS), de.shape[0])
    if slots is not None:
        Q, out = torch.from_numpy(slots).to(dev, torch.float16), []
        for i in range(0, Q.shape[0], 64):
            b = Q[i:i + 64]
            idx = torch.topk(b.reshape(-1, b.shape[-1]) @ D, k=Kmax, dim=-1).indices.reshape(b.shape[0], b.shape[1], Kmax).cpu().numpy()
            out += [np.array(merge_round_robin(idx[j], Kmax))[:Kmax] for j in range(b.shape[0])]
        topk = np.stack(out)
    else:
        Q = torch.from_numpy(qe).to(dev, torch.float16)
        topk = np.concatenate([torch.topk(Q[i:i + 256] @ D, k=Kmax, dim=1).indices.cpu().numpy() for i in range(0, Q.shape[0], 256)])

    cov, per_q = {k: [] for k in KS}, {k: [] for k in KS}
    for i, q in enumerate(queries):
        for k in KS:
            got = set(topk[i][:k].tolist())
            hits = [any(did2row.get(g) in got for g in e["gold"]) for e in qrels[q["uid"]]]
            cov[k] += hits
            per_q[k].append(sum(hits) / len(hits) if hits else 0.0)
    print(f"[{a.data} | {tag}] questions {len(queries):,} perspectives {len(cov[KS[0]]):,} corpus {len(corpus):,}")
    for k in KS:
        print(f"  Hard Coverage@{k:<3} perspective {100 * np.mean(cov[k]):5.2f}%   per-question {100 * np.mean(per_q[k]):5.2f}%")
    out = d / f"results_{tag}{'_distract' if a.distractors else ''}.json"
    out.write_text(json.dumps({"backend": tag, "data": a.data, "perspective": {k: float(np.mean(cov[k])) for k in KS},
                               "per_question": {k: float(np.mean(per_q[k])) for k in KS}, "n_questions": len(queries),
                               "n_perspectives": len(cov[KS[0]]), "n_docs": int(de.shape[0])}, indent=1))
    print("  →", out, flush=True)


if __name__ == "__main__":
    main()
