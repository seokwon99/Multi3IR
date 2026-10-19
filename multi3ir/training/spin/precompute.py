"""Build the SPIN training cache, one subcommand per step.

  hl       cache h_l at the injection layer (+ perspective targets, qwen) -> shard_NN.pt, layer.txt, instruction.txt
  repack   shard list -> flat arrays (index.npz, h_l.u16.npy, targets.f32.npy): one open instead of 300k zip records
  queries  eps-free query embeddings from a cache, cache order -> npz (raw reference line; input to mine.py)
  targets  re-encode the perspective targets of the qwen cache under an instruction (sharded, then --merge)
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi3ir import paths as _P

LAYER = {"qwen": 18, "mmembed": 18, "gme": 14}
INSTRUCTION = "Retrieve images or text relevant to the user's query."


# ── hl ──────────────────────────────────────────────────────────────────────────────────────
def _slice(h_b, attn_b):
    return [(h_b[j, :int(attn_b[j].sum())], attn_b[j, :int(attn_b[j].sum())]) for j in range(h_b.shape[0])]


def build_qwen(a):
    from multi3ir.training.backbones import qwen
    emb = qwen.load_embedder(max_length=a.max_length)
    enc = qwen.Qwen3VLTextSplitEncoder(emb).cuda().eval()
    tok_q, tok_p = qwen.make_text_tokenizer(emb, a.instruction), qwen.make_text_tokenizer(emb)

    def encode(texts):
        return _slice(*enc.encode_to_layer({k: v.cuda() for k, v in tok_q(texts).items()}, a.layer))

    def targets(persps):
        return torch.cat([enc.encode_full({k: v.cuda() for k, v in tok_p(persps[s:s + a.persp_batch]).items()})
                          .cpu().float() for s in range(0, len(persps), a.persp_batch)])
    return encode, targets


def build_mmembed(a):
    from transformers import AutoModel
    from multi3ir.training.backbones import mmembed
    m = AutoModel.from_pretrained(mmembed.MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                  attn_implementation="sdpa").cuda().eval()
    enc, tok = mmembed.MMEmbedTextSplitEncoder(m), mmembed.make_text_tokenizer(m, a.max_length)
    return lambda texts: _slice(*enc.encode_to_layer({k: v.cuda() for k, v in tok(texts).items()}, a.layer)), None


def build_gme(a):
    """Raw forward with output_hidden_states (hidden_states[L] = input to layer L); GME pads LEFT."""
    from transformers import AutoTokenizer
    from multi3ir.training.backbones import gme
    model, _ = gme.load_model()
    tok = AutoTokenizer.from_pretrained(gme.MODEL, trust_remote_code=True)
    base = model if hasattr(model, "layers") else model.model

    def encode(texts):
        inp = tok([gme.PROMPT.format(instr=a.instruction, q=t) for t in texts], padding="longest",
                  truncation=True, max_length=a.max_length, return_tensors="pt")
        inp = {k: v.cuda() for k, v in inp.items()}
        pos = gme.positions_from_mask(inp["attention_mask"])
        out = base(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                   position_ids=pos[None].expand(3, -1, -1).contiguous(), output_hidden_states=True,
                   use_cache=False, return_dict=True)
        return [(out.hidden_states[a.layer][j][mask], torch.ones(int(mask.sum()), dtype=torch.long))
                for j, mask in enumerate(inp["attention_mask"].bool())]
    return encode, None


def _sidecar(path: Path, value: str):
    if path.exists() and path.read_text().strip() != value.strip():
        raise RuntimeError(f"{path} holds {path.read_text()!r}, this shard has {value!r}")
    path.write_text(value)


def cmd_hl(a):
    if a.with_targets and a.backend != "qwen":
        raise ValueError("--with-targets is implemented for the qwen split encoder only")
    a.layer = LAYER[a.backend] if a.layer is None else a.layer
    out = Path(a.out)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} exists", flush=True)
        return
    uid2query, uid2persps, ref_targets = {}, {}, {}
    with open(a.src) as f:
        for line in f:
            d = json.loads(line)
            if d["uid"] in uid2query:
                continue
            if a.with_targets:
                if len(d["positives"]) < a.min_persp:
                    continue
                uid2persps[d["uid"]] = d["positives"][:a.max_persp]
            uid2query[d["uid"]] = d["query"]
    if a.uids_pt:   # drop-in replacement for a reference shard: its uid ORDER and its targets
        if a.num_shards > 1:
            raise ValueError("--uids-pt fixes the shard's uid set; do not combine with --num-shards")
        ref = torch.load(a.uids_pt, map_location="cpu", weights_only=False, mmap=True)
        uids = [it["uid"] for it in ref]
        ref_targets = {it["uid"]: it["targets"] for it in ref if "targets" in it}
        if missing := [u for u in uids if u not in uid2query]:
            raise RuntimeError(f"{len(missing)} uids of {a.uids_pt} missing in {a.src}, e.g. {missing[:3]}")
    else:
        uids = [u for i, u in enumerate(sorted(uid2query)) if i % a.num_shards == a.shard_idx]
    print(f"[shard {a.shard_idx}/{a.num_shards}] {len(uids)} uids", flush=True)
    encode, targets = {"qwen": build_qwen, "mmembed": build_mmembed, "gme": build_gme}[a.backend](a)

    items, t0 = [], time.time()
    with torch.inference_mode():
        for s in range(0, len(uids), a.batch_size):
            batch = uids[s:s + a.batch_size]
            for u, (h, attn) in zip(batch, encode([uid2query[u] for u in batch])):
                it = {"uid": u, "h_l": h.to("cpu", torch.bfloat16).clone(), "attn": attn.to("cpu", torch.long).clone()}
                if a.with_targets:
                    it["targets"] = targets(uid2persps[u])
                elif u in ref_targets:
                    it["targets"] = ref_targets[u]
                items.append(it)
            if (s // a.batch_size) % 50 == 0:
                print(f"  {s + len(batch)}/{len(uids)}  Δt={time.time()-t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    _sidecar(out.parent / "layer.txt", f"{a.layer}\n")
    if a.backend != "mmembed":
        _sidecar(out.parent / "instruction.txt", a.instruction)   # the instruction travels with the h_l it shaped
    torch.save(items, str(out) + ".tmp")
    Path(str(out) + ".tmp").replace(out)
    print(f"[saved] {len(items)} -> {out}", flush=True)


# ── repack ──────────────────────────────────────────────────────────────────────────────────
def cmd_repack(a):
    files = sorted(Path(a.cache_dir).glob("shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"no shard_*.pt under {a.cache_dir}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    uids, T, P, D = [], [], [], None
    for f in files:                       # pass 1: shapes only (mmap faults no data pages)
        for it in torch.load(str(f), map_location="cpu", weights_only=False, mmap=True):
            uids.append(it["uid"]); T.append(it["h_l"].shape[0]); P.append(it["targets"].shape[0])
            D = it["h_l"].shape[1]
    oT, oP = np.concatenate([[0], np.cumsum(T)]), np.concatenate([[0], np.cumsum(P)])
    print(f"[scan] {len(uids):,} queries, {oT[-1]:,} tokens, {oP[-1]:,} perspectives, D={D}", flush=True)
    hl = np.lib.format.open_memmap(str(out / "h_l.u16.npy.tmp"), mode="w+", dtype=np.uint16, shape=(int(oT[-1]), D))
    tg = np.lib.format.open_memmap(str(out / "targets.f32.npy.tmp"), mode="w+", dtype=np.float32, shape=(int(oP[-1]), D))
    n = 0
    for f in files:                       # pass 2: copy in load order, so row i is the same query either way
        for it in torch.load(str(f), map_location="cpu", weights_only=False, mmap=True):
            if not bool((it["attn"] == 1).all()) or it["h_l"].dtype is not torch.bfloat16:
                raise RuntimeError(f"item {n}: attn not all-ones or h_l not bf16; cannot drop attn")
            hl[oT[n]:oT[n + 1]] = it["h_l"].contiguous().view(torch.uint16).numpy()
            tg[oP[n]:oP[n + 1]] = it["targets"].numpy()
            n += 1
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        print(f"[copy] {f.name}: {n:,}/{len(uids):,}", flush=True)
    hl.flush(); tg.flush(); del hl, tg
    os.replace(out / "h_l.u16.npy.tmp", out / "h_l.u16.npy")
    os.replace(out / "targets.f32.npy.tmp", out / "targets.f32.npy")
    layer = int((Path(a.cache_dir) / "layer.txt").read_text())
    np.savez(out / "index.npz", uids=np.array(uids), offsets_T=oT, offsets_P=oP, dim=D, layer=layer,
             source=np.array([f.name for f in files]))
    instr = Path(a.cache_dir) / "instruction.txt"
    if instr.exists():
        (out / "instruction.txt").write_text(instr.read_text())
    print(f"[saved] {out} (layer {layer})", flush=True)


# ── queries ─────────────────────────────────────────────────────────────────────────────────
def cmd_queries(a):
    from multi3ir.training.spin import data, encode, train
    cache = train.ref_cache(a.encoder, a.data)
    encoder, _ = encode.build(a.encoder, train.cache_instruction(cache, a.encoder))
    encoder.eval()
    items = data.load(cache)
    n, D = len(items), items[0]["h_l"].shape[1]
    lengths = np.array([items[i]["h_l"].shape[0] for i in range(n)])
    order = np.argsort(lengths, kind="stable")          # length-sorted batches, written back to cache order
    out = np.empty((n, D), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n, a.batch_size):
            idx = order[s:s + a.batch_size]
            T = int(lengths[idx].max())
            h = torch.zeros(len(idx), T, D, dtype=torch.bfloat16, device="cuda")
            attn = torch.zeros(len(idx), T, dtype=torch.long, device="cuda")
            for j, i in enumerate(idx):
                h[j, :lengths[i]] = items[int(i)]["h_l"].to("cuda", torch.bfloat16)
                attn[j, :lengths[i]] = 1
            out[idx] = encoder.encode_from_layer(h, attn, items.layer).float().cpu().numpy()
            if s % (a.batch_size * 100) == 0:
                print(f"  {s + len(idx):,}/{n:,}", flush=True)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite query embedding produced")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, ids=items.uids, embeddings=out)
    print(f"[saved] {out.shape} -> {a.out}", flush=True)


# ── targets ─────────────────────────────────────────────────────────────────────────────────
def cmd_targets(a):
    from multi3ir.training.spin import encode, train
    idx = np.load(train.cache_path("qwen", 18) / "index.npz", allow_pickle=True)
    uids, oP = [str(u) for u in idx["uids"]], idx["offsets_P"]
    src = {}
    with open(_P.TRAIN_QUERIES) as f:
        for line in f:
            d = json.loads(line)
            src[d["uid"]] = d["positives"][:8]
    rows = [src[u] for u in uids]
    if [len(r) for r in rows] != np.diff(oP).tolist():
        raise RuntimeError("perspective counts differ between the cache and queries.jsonl")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _sidecar(out / "instruction.txt", a.instruction)
    bounds = [len(rows) * i // a.num_shards for i in range(a.num_shards + 1)]
    if a.merge:
        merged = np.lib.format.open_memmap(str(out / "targets.f32.npy.tmp"), mode="w+", dtype=np.float32,
                                           shape=(int(oP[-1]), int(idx["dim"])))
        for s in range(a.num_shards):
            part = np.load(out / f"part_{s:02d}.npy", mmap_mode="r")
            lo, hi = int(oP[bounds[s]]), int(oP[bounds[s + 1]])
            if part.shape[0] != hi - lo:
                raise RuntimeError(f"part_{s:02d}: {part.shape[0]} rows, expected {hi - lo}")
            merged[lo:hi] = part
        merged.flush(); del merged
        (out / "targets.f32.npy.tmp").rename(out / "targets.f32.npy")
        print(f"[merged] -> {out / 'targets.f32.npy'}", flush=True)
        return
    texts = [t for r in rows[bounds[a.shard]:bounds[a.shard + 1]] for t in r]
    enc, tok = encode.build("qwen", a.instruction)
    enc.eval()
    vecs = []
    with torch.inference_mode():
        for s in range(0, len(texts), a.batch_size):
            inputs = {k: v.cuda() for k, v in tok(texts[s:s + a.batch_size]).items()}
            vecs.append(F.normalize(enc.encode_full(inputs).float(), dim=-1).cpu().numpy())
    arr = np.concatenate(vecs).astype(np.float32)
    np.save(out / f"part_{a.shard:02d}.tmp.npy", arr)
    (out / f"part_{a.shard:02d}.tmp.npy").rename(out / f"part_{a.shard:02d}.npy")
    print(f"[saved] part_{a.shard:02d} {arr.shape}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("hl")
    s.add_argument("--backend", required=True, choices=sorted(LAYER))
    s.add_argument("--src", default=str(_P.TRAIN_QUERIES), help="queries jsonl (uid, query, positives)")
    s.add_argument("--out", required=True, help="shard_NN.pt")
    s.add_argument("--layer", type=int, default=None, help="default: 18 qwen/mmembed, 14 gme")
    s.add_argument("--shard-idx", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--max-length", type=int, default=256)
    s.add_argument("--instruction", default=INSTRUCTION)
    s.add_argument("--with-targets", action="store_true", help="also encode each uid's perspectives (qwen)")
    s.add_argument("--uids-pt", default=None, help="reference shard: reuse its uid order and its targets")
    s.add_argument("--persp-batch", type=int, default=8)
    s.add_argument("--max-persp", type=int, default=8)
    s.add_argument("--min-persp", type=int, default=2)
    s.set_defaults(fn=cmd_hl)
    s = sub.add_parser("repack")
    s.add_argument("--cache-dir", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_repack)
    s = sub.add_parser("queries")
    s.add_argument("--encoder", default="qwen", choices=sorted(LAYER))
    s.add_argument("--data", default="full")
    s.add_argument("--out", required=True)
    s.add_argument("--batch-size", type=int, default=16)
    s.set_defaults(fn=cmd_queries)
    s = sub.add_parser("targets")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--num-shards", type=int, required=True)
    s.add_argument("--shard", type=int)
    s.add_argument("--merge", action="store_true")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--instruction", default=INSTRUCTION)
    s.set_defaults(fn=cmd_targets)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
