"""Encode text with a frozen backbone → npz {ids, embeddings}.

  --field query        + --layout flat     ids=uid,       embeddings (Q, D)     naive queries
  --field perspectives + --layout grouped  ids=uid,       embeddings (Q, m, D)  LLM expansion
  --field perspectives + --layout flat     ids='uid|idx', embeddings (N, D)     oracle

  python -m multi3ir.eval.encode --backend qwen --field query --src $BENCH/eval/queries_1039.jsonl --out q.npz
"""
import argparse
import json
import os

import numpy as np

from multi3ir import paths as _P

INSTRUCTION = "Retrieve images or text relevant to the user's query."


def read_rows(src, field, layout):
    """-> (ids, texts, m); m is the fixed perspective count for layout=grouped, else None."""
    rows = [json.loads(ln) for ln in open(src) if ln.strip()]
    if field == "query":
        by_uid = {r["uid"]: r["query"] for r in rows}
        ids = sorted(by_uid)
        return ids, [by_uid[u] for u in ids], None
    listed = [r for r in rows if isinstance(r.get("perspectives"), list)]
    if layout == "grouped":
        if not listed:
            raise ValueError("--layout grouped requires rows with a 'perspectives' list")
        m = len(listed[0]["perspectives"])
        bad = [r["uid"] for r in listed if len(r["perspectives"]) != m]
        if bad:
            raise ValueError(f"--layout grouped needs a fixed perspective count; {len(bad)} rows differ from {m}")
        return [r["uid"] for r in listed], [t for r in listed for t in r["perspectives"]], m
    if listed:
        pairs = [(f'{r["uid"]}|{i}', t) for r in listed for i, t in enumerate(r["perspectives"])]
    else:  # one perspective per row
        pairs = [(f'{r["uid"]}|{r["perspective_idx"]}', r["summary"]) for r in rows]
    return [p[0] for p in pairs], [p[1] for p in pairs], None


def _load_state(model, ckpt):
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck["state"] if "state" in ck else ck, strict=False)
    print(f"[ckpt] {ckpt} (missing={len(miss)}, unexpected={len(unexp)})")


def encode_qwen(texts, args):
    import torch, tqdm
    from multi3ir.training.backbones.qwen import load_embedder, make_text_tokenizer, pool_last
    embedder = load_embedder(max_length=args.max_length)
    if args.ckpt:
        _load_state(embedder.model, args.ckpt)
    embedder.model = embedder.model.cuda().eval()
    tok = make_text_tokenizer(embedder, instruction=args.instruction)
    out = []
    with torch.no_grad():
        for s in tqdm.tqdm(range(0, len(texts), args.batch), desc="enc"):
            inputs = {k: v.to("cuda") for k, v in tok(texts[s:s + args.batch]).items()}
            h = embedder.model(**inputs).last_hidden_state
            out.append(pool_last(h, inputs["attention_mask"]).float().cpu().numpy())
    return np.concatenate(out)


def encode_mmembed(texts, args):
    import torch, tqdm
    from transformers import AutoModel
    model = AutoModel.from_pretrained("nvidia/MM-Embed", trust_remote_code=True,
                                      torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.ckpt:
        _load_state(model, args.ckpt)
    model = model.cuda().eval()
    out = []
    with torch.no_grad():
        for s in tqdm.tqdm(range(0, len(texts), args.batch), desc="enc"):
            e = model.encode([{"txt": t} for t in texts[s:s + args.batch]], is_query=True,
                             instruction=args.instruction, max_length=args.max_length)["hidden_states"]
            out.append(e.cpu().float().numpy())
            torch.cuda.empty_cache()
    return np.concatenate(out)


def encode_gme(texts, args):
    """GME through its own transformers API (what the corpus was encoded with). Needs the gme venv."""
    if args.ckpt:
        raise ValueError("--ckpt is not supported for gme")
    import transformers.modeling_rope_utils as ru
    ru.ROPE_INIT_FUNCTIONS.setdefault("mrope", ru.ROPE_INIT_FUNCTIONS["default"])
    if hasattr(ru, "ROPE_VALIDATION_FUNCTIONS"):
        ru.ROPE_VALIDATION_FUNCTIONS.setdefault("mrope", ru.ROPE_VALIDATION_FUNCTIONS["default"])
    import torch, tqdm
    from transformers import AutoModel
    model = AutoModel.from_pretrained("Alibaba-NLP/gme-Qwen2-VL-7B-Instruct", trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).to("cuda").eval()
    out = []
    with torch.no_grad():
        for s in tqdm.tqdm(range(0, len(texts), args.batch), desc="enc"):
            e = model.get_text_embeddings(texts=texts[s:s + args.batch], instruction=args.instruction)
            out.append(e.cpu().float().numpy())
            torch.cuda.empty_cache()
    return np.concatenate(out)


BACKENDS = {"qwen": encode_qwen, "mmembed": encode_mmembed, "gme": encode_gme}
MAX_LENGTH = {"qwen": 256, "mmembed": 512, "gme": 4096}


def encode(backend, texts, instruction=INSTRUCTION, batch=8, ckpt=None, max_length=None):
    """Unit-normalised (N, D) float32 embeddings; max_length None = the backbone's own default."""
    args = argparse.Namespace(instruction=instruction, batch=batch, ckpt=ckpt, max_length=max_length)
    emb = np.asarray(BACKENDS[backend](texts, args), dtype=np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    if not (n > 1e-6).all():
        raise RuntimeError(f"{int((n <= 1e-6).sum())} zero-norm embeddings")
    return emb / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--src", required=True, help="input jsonl")
    ap.add_argument("--out", required=True, help="output npz")
    ap.add_argument("--field", choices=["query", "perspectives"], default="query")
    ap.add_argument("--layout", choices=["flat", "grouped"], default="flat")
    ap.add_argument("--ckpt", default=None, help="fine-tuned state dict to load over the base (qwen/mmembed)")
    ap.add_argument("--instruction", default=INSTRUCTION)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=None, help="default: 256 qwen / 512 mmembed / 4096 gme")
    args = ap.parse_args()

    ids, texts, m = read_rows(args.src, args.field, args.layout)
    print(f"[src] {len(ids)} ids, {len(texts)} texts" + (f", m={m}" if m else "") + f" ← {args.src}")
    emb = encode(args.backend, texts, args.instruction, args.batch, args.ckpt, args.max_length or MAX_LENGTH[args.backend])
    if m:
        emb = emb.reshape(len(ids), m, -1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, ids=np.array(ids), embeddings=emb)
    print(f"[saved] {emb.shape} → {args.out}")


if __name__ == "__main__":
    main()
