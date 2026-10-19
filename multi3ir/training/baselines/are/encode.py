"""Apply a trained ARE checkpoint to queries -> (N, K, D) npz for eval.run_eval --query-type spin.

    python -m multi3ir.training.baselines.are.encode --ckpt results/ckpts/<run>/step005000.ckpt --K 5 --out are_q.npz
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi3ir import paths as _P
from multi3ir.training.baselines.are.model import ARE, format_queries, get_tokenizer


def restore(model: ARE, ckpt):
    """Load the trainable tensors of a Lightning checkpoint (keys prefixed 'model.')."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    own = dict(model.named_parameters())
    n, unmatched = 0, []
    for k, v in sd.items():
        key = k[len("model."):] if k.startswith("model.") else k
        if key in own:
            own[key].data.copy_(v.to(own[key].dtype)); n += 1
        else:
            unmatched.append(key)
    if unmatched:
        raise RuntimeError(f"{ckpt}: {len(unmatched)} tensors do not fit this model, e.g. {unmatched[:3]}")
    print(f"[ARE] loaded {n} tensors from {ckpt}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--backbone", choices=["qwen", "gme"], default="qwen")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--queries", default=str(_P.EVAL_QUERIES))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--instruction", default=None, help="override the backbone's default query instruction")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.queries)]
    uids, queries = [str(r["uid"]) for r in rows], [r["query"] for r in rows]
    model = ARE(backbone=args.backbone, lora_r=args.lora_r).cuda()
    restore(model, args.ckpt)
    tok = get_tokenizer(model)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(queries), args.batch_size):
            t = tok(format_queries(model, queries[i:i + args.batch_size], args.instruction), padding=True,
                    truncation=True, max_length=256, return_tensors="pt")
            out.append(model.forward_infer(t["input_ids"].cuda(), t["attention_mask"].cuda(), args.K).float().cpu().numpy())
            if (i // args.batch_size) % 20 == 0:
                print(f"  {i + len(out[-1])}/{len(queries)}", flush=True)
    emb = np.concatenate(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=np.array(uids), embeddings=emb.astype(np.float16))
    print(f"[saved] {emb.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
