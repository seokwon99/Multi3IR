"""Backbone selection, and: apply a trained SPIN checkpoint to queries -> (N, m, D) npz.

    python -m multi3ir.training.spin.encode --encoder qwen --ckpt results/ckpts/spin/<run>/spin_best.pt --out q.npz
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi3ir import paths as _P
from multi3ir.training.spin import model as spin_model

ENCODERS = ("qwen", "mmembed", "gme")


def build(name: str, instruction: str | None):
    """-> (split encoder, tokenize). qwen/gme take the query instruction the cache was built with;
    mmembed has a fixed one and refuses an override."""
    if (instruction is not None) != (name in ("qwen", "gme")):
        raise ValueError(f"encoder {name!r} {'needs' if name != 'mmembed' else 'takes no'} instruction (got {instruction!r})")
    if name == "qwen":
        from multi3ir.training.backbones import qwen
        emb = qwen.load_embedder()
        return qwen.Qwen3VLTextSplitEncoder(emb), qwen.make_text_tokenizer(emb, instruction)
    if name == "mmembed":
        from transformers import AutoModel
        from multi3ir.training.backbones import mmembed
        m = AutoModel.from_pretrained(mmembed.MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa").cuda().eval()
        return mmembed.MMEmbedTextSplitEncoder(m), mmembed.make_text_tokenizer(m)
    if name == "gme":
        from multi3ir.training.backbones import gme
        m, proc = gme.load_model()
        return gme.GmeTextSplitEncoder(m), gme.make_text_tokenizer(proc, instruction=instruction)
    raise ValueError(f"unknown encoder {name!r}; expected one of {ENCODERS}")


def ckpt_instruction(ckpt: str, encoder: str) -> str | None:
    """The query instruction a run was trained under (recorded by train.py in args.json)."""
    if encoder == "mmembed":
        return None
    cfg = Path(ckpt).parent / "args.json"
    instr = json.loads(cfg.read_text()).get("instruction") if cfg.exists() else None
    if instr is None:
        raise RuntimeError(f"{cfg} records no instruction; pass --instruction (pre-2026-08-29 {encoder} runs "
                           f"used backbones.{encoder}.INSTRUCTION)")
    return instr


def read_queries(path) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.setdefault(d["uid"], d["query"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="qwen", choices=ENCODERS)
    p.add_argument("--queries-jsonl", default=str(_P.EVAL_QUERIES))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--instruction", default=None, help="override the instruction recorded in args.json")
    p.add_argument("--limit", type=int, default=0, help="encode only the first N queries (smoke tests)")
    args = p.parse_args()

    instruction = args.instruction or ckpt_instruction(args.ckpt, args.encoder)
    print(f"[instruction] {instruction!r}", flush=True)
    encoder, tokenize = build(args.encoder, instruction)
    encoder.eval()
    spin = spin_model.load(args.ckpt, encoder, args.encoder)
    print(f"[model] m={spin.m} layer={spin.layer} rank={spin.rank}", flush=True)

    uid2query = read_queries(args.queries_jsonl)
    uids = list(uid2query)[:args.limit or None]
    out = []
    with torch.no_grad():
        for s in range(0, len(uids), args.batch_size):
            chunk = uids[s:s + args.batch_size]
            inputs = {k: v.to(encoder.device) for k, v in tokenize([uid2query[u] for u in chunk]).items()}
            h, attn = encoder.encode_to_layer(inputs, spin.layer)
            for j in range(len(chunk)):                    # one query at a time, padding stripped, as in training
                keep = attn[j].bool()
                slots = F.normalize(spin(h[j:j + 1, keep], attn[j:j + 1, keep]).float(), dim=-1)   # (m, 1, D)
                out.append(slots.permute(1, 0, 2).cpu().numpy())
            if s % (args.batch_size * 25) == 0:
                print(f"  {s + len(chunk)}/{len(uids)}", flush=True)
    emb = np.concatenate(out).astype("float32")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=np.array(uids), embeddings=emb)
    print(f"[saved] {emb.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
