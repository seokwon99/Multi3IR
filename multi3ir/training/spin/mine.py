"""Mine hard negatives once per question from the eps-free query embeddings (precompute queries).

  --anchor persp     rank all other questions' perspectives by similarity to the query
  --anchor question  rank questions by similarity, then take their perspectives in that order
Output: indices into the cache's flat perspective array, k per question, in ranking order.

    python -m multi3ir.training.spin.mine --encoder qwen --data full --anchor question \\
        --queries results/ckpts/spin_mining/qwen_full_queries.npz --out results/ckpts/spin_mining/qwen_full_question_k2048.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi3ir.training.spin import data, encode, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="qwen", choices=encode.ENCODERS)
    p.add_argument("--data", default="full", choices=train.DATA_CHOICES)
    p.add_argument("--queries", required=True, help="npz from precompute queries")
    p.add_argument("--anchor", required=True, choices=["persp", "question"])
    p.add_argument("--k", type=int, default=train.NEG_DEPTH)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    cache = data.load(train.ref_cache(a.encoder, a.data))
    counts, flat = cache.counts(), cache.all_targets()
    gid_of = {u: i for i, u in enumerate(dict.fromkeys(cache.uids))}   # the cache repeats some questions
    row_gid = torch.from_numpy(np.repeat([gid_of[u] for u in cache.uids], counts)).cuda()
    query_gid = torch.from_numpy(np.array([gid_of[u] for u in cache.uids])).cuda()
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    bank = torch.empty(flat.shape, device="cuda", dtype=torch.float16)
    for s in range(0, flat.shape[0], 65536):
        bank[s:s + 65536] = F.normalize(flat[s:s + 65536].cuda().float(), dim=-1).half()
    P = bank.shape[0]
    print(f"[bank] {P:,} perspectives, {len(gid_of):,} distinct questions", flush=True)

    qz = np.load(a.queries, allow_pickle=True)
    if not (np.array([str(u) for u in qz["ids"]]) == cache.uids).all():
        raise RuntimeError(f"{a.queries} is not in cache order")
    Q = F.normalize(torch.from_numpy(qz["embeddings"]).cuda().float(), dim=-1).half()
    N = Q.shape[0]
    neg_idx, neg_cos = np.empty((N, a.k), np.int32), np.empty((N, a.k), np.float16)
    for s in range(0, N, a.chunk):
        q = Q[s:s + a.chunk]
        sim_p = (q @ bank.t()).float()
        own = row_gid[None] == query_gid[s:s + q.shape[0], None]     # own perspectives are the positives
        sim_p[own] = -2.0
        if a.anchor == "persp":
            if int((~own).sum(1).min()) < a.k:
                raise RuntimeError(f"fewer than k={a.k} candidates for a query; lower --k")
            top = torch.topk(sim_p, a.k, dim=1)
            neg_idx[s:s + q.shape[0]], neg_cos[s:s + q.shape[0]] = top.indices.cpu().numpy(), top.values.half().cpu().numpy()
        else:
            sim_q = (q @ Q.t()).float()
            sim_q[query_gid[None] == query_gid[s:s + q.shape[0], None]] = -2.0
            order, keep, scores = torch.argsort(sim_q, 1, descending=True).cpu().numpy(), (~own).cpu().numpy(), sim_p.cpu().numpy()
            for r in range(q.shape[0]):
                picked = []
                for row in order[r]:
                    st, c = int(starts[row]), int(counts[row])
                    picked.extend(j for j in range(st, st + c) if keep[r, j])
                    if len(picked) >= a.k:
                        break
                if len(picked) < a.k:
                    raise RuntimeError(f"query {s + r} gathered {len(picked)} < k={a.k} perspectives; lower --k")
                neg_idx[s + r], neg_cos[s + r] = picked[:a.k], scores[r, picked[:a.k]]
        if s % (a.chunk * 20) == 0:
            print(f"  {s + q.shape[0]:,}/{N:,}", flush=True)
    np.savez(a.out, neg_idx=neg_idx, neg_cos=neg_cos, ids=qz["ids"], anchor=a.anchor, k=a.k,
             data=a.data, encoder=a.encoder, n_perspectives=P)
    print(f"[saved] {neg_idx.shape} -> {a.out}; top-128 cosine mean {neg_cos[:, :128].astype(np.float32).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
