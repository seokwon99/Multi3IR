"""LM-judge BeRDS retrievals (cached per (doc, perspective) pair), then MRecall@k / Precision@k or SC@k.

  python -m multi3ir.eval.berds.judge --encoder qwen3vl-raw                 # OPENAI_API_KEY; hc = BeRDS paper metric
  python -m multi3ir.eval.berds.judge --encoder qwen3vl-raw --variant sc    # strict 'fully supports' -> SC@k
  python -m multi3ir.eval.berds.judge --encoder qwen3vl-raw --score-only
"""
import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from multi3ir.eval.berds.data import DOMAINS, ROOT

log = logging.getLogger("berds.judge")
HC_PROMPT = """You are evaluating whether a passage expresses a given perspective.

A perspective is a declarative claim or stance on a subjective question.
A passage "expresses" the perspective if it argues for, supports, or directly states the perspective. Mere mention without endorsement is NOT enough. Mention of the opposite stance is NOT enough.

Passage:
{doc}

Perspective:
{perspective}

Does the passage express the perspective above? Answer with exactly "yes" or "no"."""
SC_PROMPT = """You are evaluating whether a passage FULLY supports a given perspective.

A perspective is a declarative claim or stance on a subjective question.
A passage "fully supports" the perspective if it clearly states, argues for, or directly endorses the claim.
Mere topical mention, opposing stance, or partial relevance is NOT enough — answer "no" in those cases.

Passage:
{doc}

Perspective:
{perspective}

Does the passage fully support the perspective? Answer with exactly "yes" or "no"."""
# criterion -> (prompt, cache file, judged-file suffix, top_k, workers); the two label sets are not interchangeable
VARIANTS = {"hc": (HC_PROMPT, "judge_cache.jsonl", "judged", 10, 20),
            "sc": (SC_PROMPT, "judge_cache_soft.jsonl", "soft_judged", 5, 32)}


def pair_key(doc, persp):
    return hashlib.sha256(doc.encode() + b"||" + persp.encode()).hexdigest()


def judge(args, prompt, cache_path, suffix, top_k):
    import openai
    client = openai.OpenAI()
    cache = {}
    if cache_path.exists():
        for line in cache_path.open():
            r = json.loads(line)
            cache[r["key"]] = int(r["label"])
    docs = {}
    for line in (ROOT / "corpus/corpus.jsonl").open():
        d = json.loads(line)
        docs[d["docid"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    log.info("cache %d entries, corpus %d docs", len(cache), len(docs))

    def one(doc, persp):
        r = client.chat.completions.create(model=args.model, temperature=0.0, max_tokens=4, messages=[
            {"role": "user", "content": prompt.format(doc=doc[:3000], perspective=persp)}])
        return int(r.choices[0].message.content.strip().lower().startswith("y"))

    for split in args.splits:
        for domain in args.domains:
            data = json.loads((ROOT / "runs" / args.encoder / f"{domain}.{split}.retrieval.json").read_text())
            todo = {}
            for r in data:
                for t in r["top"][:top_k]:
                    for p in r["perspectives"]:
                        if docs.get(t["docid"]):
                            k = pair_key(docs[t["docid"]], p)
                            if k not in cache:
                                todo[k] = (docs[t["docid"]], p)
            log.info("[%s/%s] %d questions, %d pairs to judge", domain, split, len(data), len(todo))
            errors, t0 = 0, time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as ex, cache_path.open("a") as fh:
                futs = {ex.submit(one, d, p): k for k, (d, p) in todo.items()}
                for n, fut in enumerate(as_completed(futs), 1):
                    try:
                        label = fut.result()
                    except Exception as e:
                        errors += 1
                        if errors < 3:
                            log.warning("judge err: %s", e)
                        continue
                    cache[futs[fut]] = label
                    fh.write(json.dumps({"key": futs[fut], "label": label}) + "\n")
                    if n % 500 == 0:
                        log.info("    %d/%d  %.1f/s  err=%d", n, len(todo), n / (time.time() - t0), errors)
            out = []
            for r in data:
                labels = [[int(cache.get(pair_key(docs[t["docid"]], p), 0)) for p in r["perspectives"]]
                          if docs.get(t["docid"]) else [0] * len(r["perspectives"]) for t in r["top"][:top_k]]
                out.append({**r, "judge_labels": labels})
            (ROOT / "runs" / args.encoder / f"{domain}.{split}.{suffix}.json").write_text(json.dumps(out))
            log.info("  judged (errors=%d)", errors)


def score(args, suffix):
    """labels[d][p] = 1 if doc d expresses perspective p. MRecall@k: every (min(m,k)) perspective hit in top-k;
    Precision@k: fraction of top-k docs expressing some perspective; SC@k: fraction of perspectives hit."""
    rows = []
    for split in args.splits:
        for domain in args.domains:
            data = json.loads((ROOT / "runs" / args.encoder / f"{domain}.{split}.{suffix}.json").read_text())
            row = {"encoder": args.encoder, "domain": domain, "split": split, "variant": args.variant, "n_questions": len(data)}
            for k in args.ks:
                L = [np.array(r["judge_labels"][:k]) for r in data]
                if args.variant == "sc":
                    row[f"SC@{k}"] = float(np.mean([(a.sum(0) >= 1).mean() if a.size else 0.0 for a in L]))
                else:
                    row[f"MRecall@{k}"] = float(np.mean([int((a.sum(0) >= 1).sum() >= min(a.shape[1], k)) if a.size else 0 for a in L]))
                    row[f"Precision@{k}"] = float(np.mean([(a.sum(1) >= 1).sum() / k if a.size else 0.0 for a in L]))
            print(f"[{domain}/{split}]  Nq={len(data):>5d}  " + "  ".join(f"{m}={v:.3f}" for m, v in row.items() if "@" in m))
            rows.append(row)
    (ROOT / "runs" / args.encoder / "metrics.json").write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="hc")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=None, help="default: 20 hc / 32 sc")
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    prompt, cache, suffix, top_k, workers = VARIANTS[args.variant]
    args.workers = args.workers or workers
    if not args.score_only:
        if "OPENAI_API_KEY" not in os.environ:
            raise SystemExit("set OPENAI_API_KEY")
        judge(args, prompt, ROOT / cache, suffix, top_k)
    score(args, suffix)


if __name__ == "__main__":
    main()
