"""PIR relevance eval (port of colinzhaoust/pir demo.ipynb): MRR, Recall@k, macro-Recall, p-Recall per domain.

  python -m multi3ir.eval.pir.evaluate --encoder qwen3vl-raw
"""
import argparse
import json
import logging
from collections import defaultdict

import numpy as np

from multi3ir.eval.pir.data import DOMAINS, ROOT, load_domain
from multi3ir.eval.pir.encode import load_encoder

log = logging.getLogger("pir.eval")
KS = (1, 5, 10)
# perspective-text substring -> query label (verbatim from process_hugginface_version.py), and the default
LABEL_RULES = {
    "ambigqa": ({"perspective": "perspective"}, "perspective"),
    "story": ({"entities": "entity", "analogy": "analogy"}, "analogy"),
    "perspectrum": ({"support": "support", "oppose": "undermine", "relates to": "general"}, "support"),
    "allsides": ({"left": "left", "right": "right", "center": "center"}, "left"),
    "exfever": ({"supports": "SUPPORT", "refutes": "REFUTE", "no information": "NOT ENOUGH INFO"}, "NOT ENOUGH INFO"),
    "agnews": ({"relates to": "subtopic", "happened in": "location"}, "subtopic"),
}


def query_label(domain, perspective):
    rules, default = LABEL_RULES[domain]
    return next((lab for needle, lab in rules.items() if needle in perspective), default)


def evaluate(scores, qids, dids, qrels, labels, roots):
    """scores (Nq, Nd). Recall@k = any positive in top-k; MRR averages RR over positives;
    macro = mean over query labels; p-Recall = mean over root queries (a root with one variant counts 0)."""
    col = {d: i for i, d in enumerate(dids)}
    hits = {k: np.zeros(len(qids), dtype=np.int32) for k in KS}
    rr = np.zeros(len(qids))
    for i, qid in enumerate(qids):
        pos = [col[d] for d in qrels.get(qid, {}) if d in col]
        if not pos:
            continue
        order = np.argsort(-scores[i], kind="stable")
        rank = {c: r for r, c in enumerate(order)}
        rr[i] = np.mean([1.0 / (rank[c] + 1) for c in pos])
        for k in KS:
            hits[k][i] = int(bool(set(order[:k].tolist()) & set(pos)))
    by_label = defaultdict(list)
    for i, qid in enumerate(qids):
        by_label[labels[qid]].append(i)
    by_root = defaultdict(list)
    for i, qid in enumerate(qids):
        by_root[roots[qid]].append(i)
    res = {"mrr": float(rr.mean())}
    for k in KS:
        res[f"recall@{k}"] = float(hits[k].mean())
        res[f"macro_recall@{k}"] = float(np.mean([hits[k][ix].mean() for ix in by_label.values()]))
        res[f"p_recall@{k}"] = float(np.mean([hits[k][ix].mean() if len(ix) > 1 else 0.0 for ix in by_root.values()]))
    res["per_label"] = {lab: {"mrr": float(rr[ix].mean()), **{f"recall@{k}": float(hits[k][ix].mean()) for k in KS}}
                        for lab, ix in by_label.items()}
    return res


def run_domain(enc, domain, out_dir):
    queries, corpus, qrels = load_domain(domain)
    qids, dids = list(queries), list(corpus)
    qtexts, dtexts = [queries[q]["text"] for q in qids], [corpus[d]["text"] for d in dids]
    log.info("[%s] Q=%d D=%d", domain, len(qids), len(dids))
    if hasattr(enc, "score_matrix"):
        scores = enc.score_matrix(qtexts, dtexts)
    else:
        scores = enc.encode_queries(qtexts) @ enc.encode_docs(dtexts).T
    scores = scores.astype(np.float32)
    np.save(out_dir / f"{domain}.scores.npy", scores)
    (out_dir / f"{domain}.qids.json").write_text(json.dumps(qids))
    (out_dir / f"{domain}.dids.json").write_text(json.dumps(dids))
    res = evaluate(scores, qids, dids, qrels,
                   {q: query_label(domain, queries[q]["meta"]["perspective"]) for q in qids},
                   {q: queries[q]["meta"]["src_query"] for q in qids})
    print(f"[{domain}]  MRR={res['mrr']:.3f}  " + "  ".join(f"R@{k}={res[f'recall@{k}']:.3f}" for k in KS))
    for lab, r in sorted(res["per_label"].items()):
        print(f"  {lab:>14s}  MRR={r['mrr']:.3f}  " + "  ".join(f"R@{k}={r[f'recall@{k}']:.3f}" for k in KS))
    print("  macro-avg              " + "  ".join(f"R@{k}={res[f'macro_recall@{k}']:.3f}" for k in KS))
    print("  p-Recall (paper)       " + "  ".join(f"@{k}={res[f'p_recall@{k}']:.3f}" for k in KS))
    return {"domain": domain, **{k: v for k, v in res.items() if k != "per_label"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["qwen3vl-raw", "qwen3vl-base", "mm-embed", "gme", "bm25"])
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base-ckpt", default=None, help="qwen3vl-base: fine-tuned state dict")
    ap.add_argument("--run-name", default=None, help="runs/<name> (default: the encoder's name)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    enc = load_encoder(args.encoder, args.device, args.base_ckpt)
    out_dir = ROOT / "runs" / (args.run_name or enc.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_domain(enc, d, out_dir) for d in args.domains]
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\n=== Summary  {out_dir.name} ===\n" + "  ".join(f"{h:>12s}" for h in ("domain", "MRR", "R@1", "R@5", "R@10", "macroR@5", "p-R@5")))
    for r in rows:
        print(f"{r['domain']:>12s}  " + "  ".join(f"{r[c]:>12.3f}" for c in ("mrr", "recall@1", "recall@5", "recall@10", "macro_recall@5", "p_recall@5")))
    print(f"{'mean p-R@5':>12s}  {np.mean([r['p_recall@5'] for r in rows]):>12.3f}  (paper Table 2: BM25 Avg.=0.358)")


if __name__ == "__main__":
    main()
