"""Encode the BeRDS corpus in shards, then merge into encodings/<encoder>/doc_embs.npy + doc_ids.json.

  python -m multi3ir.eval.berds.encode --encoder qwen3vl-raw --shard 0 --total 4     # one GPU per shard
  python -m multi3ir.eval.berds.encode --encoder qwen3vl-raw --merge
"""
import argparse
import glob
import json
import logging

import numpy as np

from multi3ir.eval.berds.data import ROOT
from multi3ir.eval.pir.encode import load_encoder

log = logging.getLogger("berds.encode")
CORPUS = ROOT / "corpus/corpus.jsonl"


def corpus_rows():
    for line in CORPUS.open():
        d = json.loads(line)
        yield d["docid"], (d.get("title", "") + " " + d.get("text", "")).strip()


def encode_shard(args):
    out_dir = ROOT / "encodings" / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shard_{args.shard:02d}_of_{args.total:02d}.npz"
    if out.exists():
        raise SystemExit(f"already encoded: {out}")
    rows = [(i, t) for j, (i, t) in enumerate(corpus_rows()) if j % args.total == args.shard]
    log.info("shard %d/%d: %d docs", args.shard, args.total, len(rows))
    enc = load_encoder(args.encoder, args.device, args.base_ckpt)
    embs = np.ascontiguousarray(enc.encode_docs([t for _, t in rows]).astype(np.float32))
    np.savez(out, ids=np.array([i for i, _ in rows]), embeddings=embs)
    log.info("saved %s %s", out, embs.shape)


def merge(args):
    d = ROOT / "encodings" / args.encoder
    shards = sorted(glob.glob(str(d / "shard_*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {d}")
    ids, embs = [], []
    for p in shards:
        z = np.load(p, allow_pickle=True)
        ids += z["ids"].tolist(); embs.append(z["embeddings"])
    embs = np.concatenate(embs)
    canonical = [i for i, _ in corpus_rows()]
    pos = {i: k for k, i in enumerate(ids)}
    missing = [i for i in canonical if i not in pos]
    if missing:
        raise RuntimeError(f"{len(missing)} corpus docs missing in shards, e.g. {missing[:3]}")
    np.save(d / "doc_embs.npy", embs[[pos[i] for i in canonical]])
    (d / "doc_ids.json").write_text(json.dumps(canonical))
    log.info("merged %d shards -> %s (%d docs)", len(shards), d, len(canonical))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["qwen3vl-raw", "qwen3vl-base", "mm-embed", "gme"])
    ap.add_argument("--shard", type=int)
    ap.add_argument("--total", type=int)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base-ckpt", default=None, help="qwen3vl-base: fine-tuned state dict")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.merge:
        merge(args)
    elif args.shard is None or args.total is None:
        ap.error("--shard and --total (or --merge)")
    else:
        encode_shard(args)


if __name__ == "__main__":
    main()
