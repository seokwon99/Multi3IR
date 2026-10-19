"""Dense retrieval on the BeRDS corpus -> runs/<encoder>/<domain>.<split>.retrieval.json.

  python -m multi3ir.eval.berds.retrieve --encoder qwen3vl-raw
"""
import argparse
import json
import logging

import numpy as np
import torch

from multi3ir.eval.berds.data import DOMAINS, ROOT, load_domain
from multi3ir.eval.pir.encode import load_encoder

log = logging.getLogger("berds.retrieve")
K_TOP = 100


def topk(q_emb, d_emb, k, chunk=200_000):
    """Chunked cosine top-k on GPU when available -> (idx, scores), each (Nq, k)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.nn.functional.normalize(torch.from_numpy(q_emb).to(dev).float(), dim=-1)
    scores = torch.full((Q.shape[0], k), -1e9, device=dev)
    idx = torch.zeros((Q.shape[0], k), device=dev, dtype=torch.long)
    for s in range(0, d_emb.shape[0], chunk):
        D = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(d_emb[s:s + chunk])).to(dev).float(), dim=-1)
        local = torch.topk(Q @ D.T, min(k, D.shape[0]), dim=1)
        merged = torch.topk(torch.cat([scores, local.values], 1), k, dim=1)
        scores = merged.values
        idx = torch.gather(torch.cat([idx, local.indices + s], 1), 1, merged.indices)
    return idx.cpu().numpy(), scores.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["qwen3vl-raw", "qwen3vl-base", "mm-embed", "gme"])
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--domains", nargs="+", default=DOMAINS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base-ckpt", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    enc_dir = ROOT / "encodings" / args.encoder
    d_emb = np.load(enc_dir / "doc_embs.npy")
    did = json.loads((enc_dir / "doc_ids.json").read_text())
    log.info("corpus: %d docs × %d", len(did), d_emb.shape[1])
    enc = load_encoder(args.encoder, args.device, args.base_ckpt)
    out_dir = ROOT / "runs" / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for domain in args.domains:
            rows = load_domain(domain, split)
            idx, sc = topk(enc.encode_queries([r["question"] for r in rows]), d_emb, K_TOP)
            results = [{"qid": r["id"], "domain": domain, "split": split, "question": r["question"],
                        "perspectives": r["perspectives"],
                        "top": [{"docid": did[int(idx[i, j])], "score": float(sc[i, j]), "rank": j + 1}
                                for j in range(K_TOP)]} for i, r in enumerate(rows)]
            out = out_dir / f"{domain}.{split}.retrieval.json"
            out.write_text(json.dumps(results))
            log.info("saved %s (%d Q)", out, len(results))


if __name__ == "__main__":
    main()
