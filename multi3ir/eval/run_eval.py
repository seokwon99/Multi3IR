"""Score a query npz on the benchmark corpus: HC@k (hard coverage) / Hit / Recall / nDCG per slot-merge (paper: round_robin).

  python -m multi3ir.eval.run_eval --query-type spin --query-npz Q.npz --tag <run> --mmr 0.2
"""
import argparse
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from multi3ir import paths as _P

KS = (5, 10, 20, 100)                  # paper: HC@k


# ── npz stores ───────────────────────────────────────────────────────────────
def sidecars(npz_path):
    """Extract an npz's members to .npy files beside it (once); npz members cannot be mmapped."""
    src = Path(npz_path)
    out = {}
    with zipfile.ZipFile(src) as z:
        for member in z.namelist():
            key = member.rsplit(".npy", 1)[0]
            dst = src.with_name(f"{src.stem}.{key}.npy")
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                tmp = dst.with_suffix(".tmp")
                with z.open(member) as r, open(tmp, "wb") as w:
                    shutil.copyfileobj(r, w, 64 << 20)
                tmp.replace(dst)
            out[key] = dst
    return out


def load_store(npz_path, keys=("ids", "embeddings")):
    """(ids, embeddings) of an npz store, memory-mapped."""
    s = sidecars(npz_path)
    return tuple(np.load(s[k], mmap_mode="r") for k in keys)


def load_gold(path, uids=None):
    """uid -> perspective_idx -> set(doc_id), image and text gold together."""
    gold = defaultdict(lambda: defaultdict(set))
    for line in open(path):
        d = json.loads(line)
        if uids and d["uid"] not in uids:
            continue
        for img in d.get("retrieved_images", []):
            gold[d["uid"]][d["perspective_idx"]].add(img["image_name"])
        for doc in d.get("retrieved_documents", []):
            gold[d["uid"]][d["perspective_idx"]].add(doc["doc_id"])
    return gold


# ── slot merges: per-slot top-K index matrix (m, K) -> one ranked list ───────
def merge_round_robin(top, k, scores=None):
    seen, merged = set(), []
    M, K = top.shape
    for col in range(K):
        for row in range(M):
            v = int(top[row, col])
            if v in seen:
                continue
            seen.add(v); merged.append(v)
            if len(merged) >= k:
                return merged
    return merged


def _best_scores(top, scores):
    best = {}
    for r in range(top.shape[0]):
        for c in range(top.shape[1]):
            v, s = int(top[r, c]), float(scores[r, c])
            if v not in best or s > best[v]:
                best[v] = s
    return best


def merge_score_max(top, k, scores):
    """Max similarity over slots, re-ranked."""
    return [v for v, _ in sorted(_best_scores(top, scores).items(), key=lambda x: -x[1])[:k]]


def merge_rrf(top, k, c=60):
    """Reciprocal rank fusion across slots."""
    acc = {}
    for r in range(top.shape[0]):
        for col in range(top.shape[1]):
            v = int(top[r, col])
            acc[v] = acc.get(v, 0.0) + 1.0 / (c + col + 1)
    return [v for v, _ in sorted(acc.items(), key=lambda x: -x[1])[:k]]


def merge_mmr(top, k, scores, doc_emb=None, lam=0.5, pool=None):
    """Greedy MMR over the slots' candidate union: best slot score minus lam * max similarity
    to what is already chosen. doc_emb (N, D) unit rows on the scores' device; None = pure score_max."""
    cands = sorted(_best_scores(top, scores).items(), key=lambda x: -x[1])[:pool or max(k * 10, 100)]
    if not cands:
        return []
    if doc_emb is None:
        return [v for v, _ in cands[:k]]
    ids = [v for v, _ in cands]
    rel = np.array([s for _, s in cands], dtype=np.float32)
    E = doc_emb[torch.as_tensor(ids, device=doc_emb.device)].float()
    sim = (E @ E.t()).cpu().numpy()
    chosen, chosen_idx, red = [], [], np.zeros(len(ids), dtype=np.float32)
    for _ in range(min(k, len(ids))):
        score = rel - lam * red
        for i in chosen_idx:
            score[i] = -1e9
        j = int(score.argmax())
        chosen.append(ids[j]); chosen_idx.append(j)
        red = np.maximum(red, sim[j])
    return chosen


# ── metrics ──────────────────────────────────────────────────────────────────
def ndcg(merged_pids, persps, all_gold, k):
    gain = lambda pid: sum(1 for pp in persps.values() if pid in pp)
    dcg = sum((2 ** gain(pid) - 1) / math.log2(j + 2) for j, pid in enumerate(merged_pids[:k]))
    ideal = sorted((gain(pid) for pid in all_gold), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(j + 2) for j, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def load_queries(npz_path, query_type, keep_uids):
    """uid -> (m_i, D) unit-normalised fp16 rows. naive: (Q,D); spin: (Q,m,D); oracle: 'uid|idx' rows."""
    d = np.load(npz_path, allow_pickle=True)
    emb, ids = d["embeddings"], d["ids"]
    if query_type == "naive":
        q = {str(u): emb[i:i + 1] for i, u in enumerate(ids) if str(u) in keep_uids}
    elif query_type == "spin":
        assert emb.ndim == 3, f"spin embeddings must be (Q,m,D), got {emb.shape}"
        q = {str(u): emb[i] for i, u in enumerate(ids) if str(u) in keep_uids}
    elif query_type == "oracle":
        groups = defaultdict(list)
        for i, raw in enumerate(ids):
            uid = str(raw).split("|", 1)[0]
            if uid in keep_uids:
                groups[uid].append(emb[i])
        q = {uid: np.stack(v) for uid, v in groups.items()}
    else:
        raise ValueError(f"unknown query_type={query_type}")
    for uid, e in q.items():
        e = e.astype(np.float16)
        q[uid] = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-3)
    ms = np.array([v.shape[0] for v in q.values()])
    print(f"[load_queries] {len(q)} uids, m avg={ms.mean():.2f} min={ms.min()} max={ms.max()}", flush=True)
    return q


def compute_metrics(queries, doc_tensor, doc_ids, gold, ks=KS, ktop=200, mmr_lambdas=(), save=None):
    """-> ({agg: {metric@k: [per-query values]}}, uids); save (list) collects the round_robin ranking rows."""
    aggs = {"round_robin": (merge_round_robin, False), "score_max": (merge_score_max, True),
            "rrf": (merge_rrf, False)}
    for lam in mmr_lambdas:
        aggs[f"mmr{lam}"] = ((lambda t, k, s, _l=lam: merge_mmr(t, k, s, doc_tensor, _l)), True)
    out = {a: defaultdict(list) for a in aggs}
    uids, Kmax = [], max(ks)
    for uid, Q in queries.items():
        if uid not in gold:
            continue
        uids.append(uid)
        sims = torch.from_numpy(Q).to(doc_tensor.device, doc_tensor.dtype) @ doc_tensor.T
        topk = torch.topk(sims, ktop, dim=-1)
        top_idx, top_scores = topk.indices.cpu().numpy(), topk.values.cpu().numpy()
        persps = gold[uid]
        all_g = set().union(*persps.values())
        for name, (fn, needs_score) in aggs.items():
            merged = fn(top_idx, Kmax, top_scores) if needs_score else fn(top_idx, Kmax)
            pids = doc_ids[np.array(merged)[:Kmax]]
            if save is not None and name == "round_robin":
                save.append({"uid": uid, "method": "round_robin",
                             "top": [{"doc_id": str(p), "rank": j + 1} for j, p in enumerate(pids)]})
            for k in ks:
                s = set(pids[:k].tolist())
                out[name][f"Hit@{k}"].append(1 if s & all_g else 0)
                out[name][f"Recall@{k}"].append(len(s & all_g) / max(1, len(all_g)))
                out[name][f"Coverage@{k}"].append(sum(1 for pp in persps.values() if s & pp) / len(persps))
                out[name][f"nDCG@{k}"].append(ndcg(pids, persps, all_g, k))
    return out, uids


def print_results(tag, out, ks=KS):
    for agg, m in out.items():
        print(f"\n=== {tag}  {agg}  fullcorpus ===\n{'k':>4}  HC      Hit     Recall  nDCG")
        for k in ks:
            print(f"{k:>4d}  " + "  ".join(f"{np.mean(m[f'{n}@{k}']):.4f}"
                                          for n in ("Coverage", "Hit", "Recall", "nDCG")))


def print_breakdown(tag, out, uids, ks=KS, agg="round_robin"):
    """Coverage@k by domain (last uid token) for one merge."""
    print(f"\n=== {tag}  {agg}  Coverage by domain ===")
    for k in ks:
        by = defaultdict(list)
        for uid, v in zip(uids, out[agg][f"Coverage@{k}"]):
            by[uid.split("_")[-1]].append(v)
        print(f"  @{k}: " + "  ".join(f"{d}={np.mean(v):.4f}(n={len(v)})" for d, v in sorted(by.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-npz", required=True)
    ap.add_argument("--query-type", choices=["naive", "spin", "oracle"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--corpus-img", default=str(_P.DOC_EMB["qwen"] / "corpus_docs.npz"))
    ap.add_argument("--corpus-txt", default=str(_P.DOC_EMB["qwen"] / "corpus_docs_text.npz"))
    ap.add_argument("--queries-jsonl", default=str(_P.EVAL_QUERIES))
    ap.add_argument("--gold", default=str(_P.EVAL_GOLD))
    ap.add_argument("--ktop", type=int, default=200)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--mmr", default=None, help="comma-separated MMR lambdas to also report, e.g. 0.2")
    ap.add_argument("--save-retrieval", default=None, help="write the round_robin top-Kmax ranking (jsonl)")
    ap.add_argument("--breakdown", action="store_true", help="also print Coverage@k per domain")
    args = ap.parse_args()

    eval_uids = {json.loads(l)["uid"] for l in open(args.queries_jsonl)}
    gold = load_gold(args.gold, eval_uids)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img_ids, img = load_store(args.corpus_img)
    txt_ids, txt = load_store(args.corpus_txt)
    doc_ids = np.concatenate([img_ids, txt_ids])
    doc_tensor = torch.cat([torch.from_numpy(np.array(img)), torch.from_numpy(np.array(txt))]).to(dev)
    doc_tensor = doc_tensor.half() if dev == "cuda" else doc_tensor.float()   # fp16 matmul is not supported on CPU
    print(f"[load_corpus] {tuple(doc_tensor.shape)} on {dev}", flush=True)
    queries = load_queries(args.query_npz, args.query_type, eval_uids)
    lambdas = tuple(float(x) for x in args.mmr.split(",")) if args.mmr else ()

    save = [] if args.save_retrieval else None
    out, uids = compute_metrics(queries, doc_tensor, doc_ids, gold, ks=tuple(args.ks), ktop=args.ktop,
                                mmr_lambdas=lambdas, save=save)
    print_results(args.tag, out, ks=tuple(args.ks))
    if args.breakdown:
        print_breakdown(args.tag, out, uids, ks=tuple(args.ks))
    if save is not None:
        p = Path(args.save_retrieval)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in save))
        print(f"[saved] {len(save)} rankings → {p}")


if __name__ == "__main__":
    main()
