"""GPT-5-mini perspectives through the OpenAI Batch API (same prompt as llm_expansion.py).

  build   eval queries -> batch input jsonl (--shots N adds N in-context examples from the training set)
  submit  upload, poll, download the batch output (OPENAI_API_KEY)
  parse   batch output -> {uid, perspectives, raw} jsonl, then llm_expansion.py encode
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

from multi3ir import paths as _P
from multi3ir.training.baselines.llm_expansion import SFT_DIR, SYS_PROMPT, dump, parse_persp

USER_TMPL = ("List {n} perspectives covering different aspects of the following question:\nQuestion: {q}\n\n"
             "Format your response as a numbered list (1. ... 2. ... etc.), one perspective per line, with no extra commentary.")


def cmd_build(a):
    shots = []
    if a.shots:   # one 5-perspective training question per category, fixed seed
        rows = [d for d in map(json.loads, open(_P.TRAIN_QUERIES)) if len(d["positives"]) == a.n]
        random.Random(a.seed).shuffle(rows)
        seen = set()
        for d in rows:
            cat = d["uid"].rsplit("_", 1)[-1]
            if cat not in seen:
                seen.add(cat); shots.append(d)
            if len(shots) == a.shots:
                break
    out = []
    for d in map(json.loads, open(_P.EVAL_QUERIES)):
        msgs = [{"role": "system", "content": SYS_PROMPT}]
        for s in shots:
            msgs += [{"role": "user", "content": USER_TMPL.format(n=a.n, q=s["query"])},
                     {"role": "assistant", "content": "\n".join(f"{i+1}. {p}" for i, p in enumerate(s["positives"][:a.n]))}]
        msgs.append({"role": "user", "content": USER_TMPL.format(n=a.n, q=d["query"])})
        out.append({"custom_id": d["uid"], "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": a.model, "messages": msgs, "reasoning_effort": "minimal", "max_completion_tokens": 600}})
    dump(a.out, out)


def cmd_submit(a):
    from openai import OpenAI
    client = OpenAI()
    f = client.files.create(file=open(a.inp, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"[submit] batch_id={batch.id}", flush=True)
    while True:
        b = client.batches.retrieve(batch.id)
        print(f"  [{time.strftime('%H:%M:%S')}] {b.status} {b.request_counts.completed}/{b.request_counts.total}", flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(a.poll_interval)
    if b.status != "completed":
        raise RuntimeError(f"batch ended with status {b.status}" + (f"; error file {b.error_file_id}" if b.error_file_id else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(client.files.content(b.output_file_id).read())
    print(f"[saved] -> {a.out}", flush=True)


def cmd_parse(a):
    rows, errors = [], 0
    for d in map(json.loads, open(a.inp)):
        try:
            text = d["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            errors += 1
            continue
        rows.append({"uid": d["custom_id"], "perspectives": parse_persp(text, a.n), "raw": text})
    print(f"[parse] {len(rows)} ok, {errors} errors", flush=True)
    dump(a.out, rows)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--out", default=str(SFT_DIR / "gpt5_batch_inputs.jsonl"))
    s.add_argument("--model", default="gpt-5-mini-2025-08-07")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--shots", type=int, default=0)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("submit")
    s.add_argument("--in", dest="inp", default=str(SFT_DIR / "gpt5_batch_inputs.jsonl"))
    s.add_argument("--out", default=str(SFT_DIR / "gpt5_batch_outputs.jsonl"))
    s.add_argument("--poll-interval", type=int, default=60)
    s.set_defaults(fn=cmd_submit)
    s = sub.add_parser("parse")
    s.add_argument("--in", dest="inp", default=str(SFT_DIR / "gpt5_batch_outputs.jsonl"))
    s.add_argument("--out", default=str(SFT_DIR / "generated_gpt5.jsonl"))
    s.add_argument("--n", type=int, default=5)
    s.set_defaults(fn=cmd_parse)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
