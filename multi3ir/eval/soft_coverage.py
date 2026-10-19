"""Soft Coverage@k: an LM judge labels (perspective, retrieved doc) pairs; labels are cached once per bench.

  python -m multi3ir.eval.soft_coverage prepare --retrievals A.jsonl B.jsonl --topk 20
  python -m multi3ir.eval.soft_coverage judge --model gpt-5-mini --workers 32      # OPENAI_API_KEY
  python -m multi3ir.eval.soft_coverage score --retrieval A.jsonl --ks 5 10 20

Retrieval jsonl rows: {"uid", "top": [{"doc_id", ...}, ...]} (run_eval --save-retrieval).
Cache rows (results/bench/result/judge_cache.jsonl): {"uid", "persp_idx", "doc_id", "label"}, label 0 | 1 | null.
"""
import argparse
import base64
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from multi3ir import paths as _P

log = logging.getLogger("multi3ir.soft_coverage")
CACHE = _P.BENCH / "result/judge_cache.jsonl"
CORPUS_TEXT = _P.BENCH / "corpus/text.jsonl"
IMAGE_DIRS = [_P.BENCH / "corpus/images", _P.STAGE3_SERPER / "serper_images_min4"]

JUDGE_PROMPT = """You are judging whether a document supports answering a specific perspective.

Follow this reasoning process:
1. Identify what the perspective specifically asks for (the subject AND the kind of answer it wants — e.g., "where to buy", "how to do", "why X", "is X true").
2. Describe what the document actually contains — only what is literally visible. Do NOT assume or infer content.
3. Compare the two and give a verdict.

Verdicts:
- fully: the document explicitly contains the specific answer the perspective asks for. If the perspective has multiple sub-questions joined by AND/commas, the document must address all of them.
- partially: the document has related information but does not fully answer the specific question (e.g., it covers the same topic but answers a different sub-question, or addresses only one of several sub-questions, or describes a different aspect than the one asked).
- not: the document is only topically related or unrelated, with no specific answer.

Format your output as:
<reason>your reasoning combining steps 1-3 in one paragraph</reason>
<verdict>fully|partially|not</verdict>

Here are examples of correct reasoning:

Example 1:
<reason>The perspective asks for a method to spatially join point data to polygon attributes in QGIS. The document shows the QGIS "Join Attributes by Location" dialog with input layer, join layer configuration, and attribute selection fields. This is the exact tool and configuration needed to perform the spatial join.</reason>
<verdict>fully</verdict>

Example 2:
<reason>The perspective asks whether Chi-Running and POSE methodologies specifically teach forefoot striking technique. The document shows a generic running form infographic comparing wrong versus correct posture, illustrating mid-foot strike and cadence targets, but no methodology names are mentioned anywhere. The form concepts are related but Chi-Running and POSE are never referenced.</reason>
<verdict>partially</verdict>

Example 3:
<reason>The perspective asks for a specific PHP substr code solution for trailing comma removal. The document shows only a blog title image reading "Remove the Last Character from a String in PHP" with a decorative clipart illustration. There is no code, no substr usage, and no actual solution visible.</reason>
<verdict>not</verdict>

Example 4:
<reason>The perspective asks where pre-made grounding pigtails can be purchased. The document explains what a pigtail connection is and gives step-by-step instructions for making one yourself by joining two ground wires with a 6-inch stripped wire. Purchasing pre-made pigtails or sources to buy them are never mentioned. The document covers the same subject (pigtails) but answers a different question (how to make) than the one asked (where to buy).</reason>
<verdict>partially</verdict>

Now judge the following:

<perspective>{perspective}</perspective>

<document>{doc}</document>"""


def parse_verdict(raw):
    """1 only for 'fully'; missing tag falls back to a substring scan."""
    m = re.search(r"<verdict>\s*(fully|partially|not)\s*</verdict>", raw or "", re.I)
    return int(m.group(1).lower() == "fully") if m else int("fully" in (raw or "").lower())


# ── cache + lookups ──────────────────────────────────────────────────────────
def load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            r = json.loads(line)
            cache[(r["uid"], int(r["persp_idx"]), r["doc_id"])] = r["label"]
    return cache


def save_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for (uid, pidx, did), lab in cache.items():
            f.write(json.dumps({"uid": uid, "persp_idx": pidx, "doc_id": did, "label": lab}, ensure_ascii=False) + "\n")
    tmp.replace(CACHE)


def load_eval_uids(path):
    return {json.loads(ln)["uid"] for ln in open(path)}


def load_perspectives(gold, eval_uids):
    """(uid, persp_idx) -> summary text."""
    out = {}
    for line in open(gold):
        d = json.loads(line)
        if d["uid"] in eval_uids and (d.get("summary") or "").strip():
            out[(d["uid"], d["perspective_idx"])] = d["summary"].strip()
    return out


def load_doc_text(corpus_text, gold):
    """doc_id -> text from the corpus file, then from gold inline snippets."""
    out = {}
    if corpus_text.exists():
        for line in corpus_text.open():
            d = json.loads(line)
            if d.get("text"):
                out[d["doc_id"]] = d["text"]
    for line in open(gold):
        for h in json.loads(line).get("retrieved_documents", []):
            if h.get("text") and h["doc_id"] not in out:
                out[h["doc_id"]] = h["text"]
    return out


def image_path(doc_id, image_dirs):
    return next((str(d / doc_id) for d in image_dirs if (d / doc_id).exists()), None)


# ── subcommands ──────────────────────────────────────────────────────────────
def cmd_prepare(args):
    eval_uids = load_eval_uids(args.queries_jsonl)
    persps = defaultdict(list)
    for uid, pidx in load_perspectives(args.gold, eval_uids):
        persps[uid].append(pidx)
    cache = load_cache()
    n0 = len(cache)
    for rp in args.retrievals:
        for line in open(rp):
            d = json.loads(line)
            for t in d["top"][:args.topk]:
                for pidx in persps.get(d["uid"], []):
                    cache.setdefault((d["uid"], pidx, t["doc_id"]), None)
    save_cache(cache)
    log.info("cache: %d → %d triples", n0, len(cache))


def _kwargs(model):
    if model.startswith("gpt-5"):
        return {"reasoning_effort": "medium", "max_completion_tokens": 2000}
    return {"temperature": 0.0, "max_tokens": 400}


def judge_text(client, model, doc, persp):
    r = client.chat.completions.create(model=model, messages=[
        {"role": "user", "content": JUDGE_PROMPT.format(doc=doc[:3000], perspective=persp)}], **_kwargs(model))
    return parse_verdict(r.choices[0].message.content)


def judge_image(client, model, path, persp, detail):
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    ext = path.lower().rsplit(".", 1)[-1]
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(ext, "jpeg")
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}", "detail": detail}},
        {"type": "text", "text": JUDGE_PROMPT.format(doc="(see attached image)", perspective=persp)}]}],
        **_kwargs(model))
    return parse_verdict(r.choices[0].message.content)


def cmd_judge(args):
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("set OPENAI_API_KEY")
    import openai
    client = openai.OpenAI()
    cache = load_cache()
    persp = load_perspectives(args.gold, load_eval_uids(args.queries_jsonl))
    doc_text = load_doc_text(Path(args.corpus_text), args.gold)
    image_dirs = [Path(d) for d in args.image_dirs]

    # judge once per unique (content, perspective) pair, then propagate to every triple
    groups = {"text": defaultdict(list), "image": defaultdict(list)}
    skipped = 0
    for (uid, pidx, did), v in cache.items():
        if v is not None or (uid, pidx) not in persp:
            continue
        if doc_text.get(did):
            groups["text"][(doc_text[did], persp[(uid, pidx)])].append((uid, pidx, did))
        elif (p := image_path(did, image_dirs)):
            groups["image"][(p, persp[(uid, pidx)])].append((uid, pidx, did))
        else:
            skipped += 1
    log.info("pending: text=%d image=%d unresolved=%d", len(groups["text"]), len(groups["image"]), skipped)
    if args.limit:
        groups["text"] = dict(list(groups["text"].items())[:args.limit])
        groups["image"] = dict(list(groups["image"].items())[:max(0, args.limit - len(groups["text"]))])

    errors = 0
    for kind, todo in groups.items():
        if not todo:
            continue
        fn = (lambda d, p: judge_text(client, args.model, d, p)) if kind == "text" else \
             (lambda d, p: judge_image(client, args.model, d, p, args.image_detail))
        done, t0 = 0, time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, d, p): (d, p) for (d, p) in todo}
            for fut in as_completed(futs):
                try:
                    label = fut.result()
                except Exception as e:
                    errors += 1
                    if errors < 3:
                        log.warning("%s err: %s", kind, e)
                    continue
                for triple in todo[futs[fut]]:
                    cache[triple] = label
                done += 1
                if done % 500 == 0:
                    save_cache(cache)
                    log.info("  %s %d/%d  %.1f/s  err=%d", kind, done, len(todo), done / (time.time() - t0), errors)
        save_cache(cache)
    log.info("done. errors=%d", errors)


def cmd_score(args):
    cache = load_cache()
    eval_uids = load_eval_uids(args.queries_jsonl)
    persps = defaultdict(set)
    for uid, pidx, _ in cache:
        if uid in eval_uids:
            persps[uid].add(pidx)
    retr = {}
    for line in open(args.retrieval):
        d = json.loads(line)
        retr[d["uid"]] = [t["doc_id"] for t in d["top"]]
    print(f"{'k':>4} {'Nq':>5} {'SC':>8} {'pending%':>10}")
    for K in args.ks:
        scs, n_lookups, n_pending = [], 0, 0
        for uid in sorted(eval_uids & persps.keys() & retr.keys()):
            covered = 0
            for p in persps[uid]:
                labels = [cache.get((uid, p, d)) for d in retr[uid][:K]]
                n_lookups += len(labels); n_pending += labels.count(None)
                covered += 1 in labels
            scs.append(covered / len(persps[uid]))
        print(f"{K:>4d} {len(scs):>5d} {np.mean(scs):>8.4f} {100 * n_pending / max(n_lookups, 1):>9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--queries-jsonl", default=str(_P.EVAL_QUERIES))
    common.add_argument("--gold", default=str(_P.EVAL_GOLD))
    sp = sub.add_parser("prepare", parents=[common])
    sp.add_argument("--retrievals", nargs="+", required=True)
    sp.add_argument("--topk", type=int, default=20)
    sp = sub.add_parser("judge", parents=[common])
    sp.add_argument("--corpus-text", default=str(CORPUS_TEXT))
    sp.add_argument("--image-dirs", nargs="*", default=[str(p) for p in IMAGE_DIRS])
    sp.add_argument("--image-detail", default="auto", choices=["auto", "low", "high"])
    sp.add_argument("--model", default="gpt-5-mini")
    sp.add_argument("--workers", type=int, default=32)
    sp.add_argument("--limit", type=int, default=0, help="judge only the first N groups (debug)")
    sp = sub.add_parser("score", parents=[common])
    sp.add_argument("--retrieval", required=True)
    sp.add_argument("--ks", nargs="+", type=int, default=[5, 10, 15, 20])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    {"prepare": cmd_prepare, "judge": cmd_judge, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
