"""Derive the benchmark corpus store for one backbone from local full-bench + gold stores.

Keeps every document that is gold for some question (train, eval and deep rounds): bench rows first, then
store-only rows. Output: document_embedding/<backbone>/corpus_docs{,_text}.npz — the store run_eval and
spin.train validation read. Users of the released benchmark build the same store with `stage4_embed`.

  python -m multi3ir.eval.corpus_variant qwen
"""
import argparse
import json

import numpy as np
import torch

from multi3ir import paths as _P
from multi3ir.eval.run_eval import load_gold
from multi3ir.training.spin.train import load_corpus

DEEP_GOLD = [_P.STAGE4 / "deep_nogold" / f for f in ("gold_text.jsonl", "gold_image.jsonl", "gold_text2.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encoder", choices=sorted(_P.DOC_EMB))
    a = ap.parse_args()
    dev, base = "cuda", _P.DOC_EMB[a.encoder]
    gold = set().union(*(s for pp in load_gold(_P.EVAL_GOLD).values() for s in pp.values()))
    gold |= {json.loads(l)["doc_id"] for l in open(_P.TRAIN_POSITIVES)}
    for f in DEEP_GOLD:
        gold |= {json.loads(l)["doc_id"] for l in open(f)}
    for bench, store, deep, out in (("docs.npz", "gold_docs.npz", "gold_deep_docs.npz", "corpus_docs.npz"),
                                    ("docs_text.npz", "gold_docs_text.npz", "gold_deep_docs_text.npz", "corpus_docs_text.npz")):
        emb_b, ids_b = load_corpus(a.encoder, dev, files=(bench,))
        emb_s, ids_s = load_corpus(a.encoder, dev, files=(store, deep))
        ids_b, ids_s = [str(d) for d in ids_b], [str(d) for d in ids_s]
        keep_b = [i for i, d in enumerate(ids_b) if d in gold]
        seen, keep_s = {ids_b[i] for i in keep_b}, []
        for i, d in enumerate(ids_s):
            if d in gold and d not in seen:
                keep_s.append(i); seen.add(d)
        emb = torch.cat([emb_b[torch.tensor(keep_b, device=dev)], emb_s[torch.tensor(keep_s, dtype=torch.long, device=dev)]])
        ids = np.array([ids_b[i] for i in keep_b] + [ids_s[i] for i in keep_s])
        assert len(set(ids)) == len(ids)
        np.savez(base / out, ids=ids, embeddings=emb.to(torch.float16).cpu().numpy())
        print(f"[saved] {base / out}: bench {len(keep_b)} + store-only {len(keep_s)} = {len(ids)}", flush=True)
        del emb_b, emb_s, emb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
