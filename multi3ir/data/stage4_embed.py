"""Stage 4d: encode documents with a backbone — queue chunks, encode on any number of GPUs, merge into an npz store.

    python -m multi3ir.data.stage4_embed queue  --model qwen3vl8b --source corpus   # the evaluation corpus (corpus/*_urls.parquet) -> chunks of 500 docs
    python -m multi3ir.data.stage4_embed encode --model qwen3vl8b --source corpus   # one GPU each, run as many as you like
    python -m multi3ir.data.stage4_embed merge  --model qwen3vl8b --source corpus   # -> document_embedding/<model>/corpus_docs{,_text}.npz
    --source deep | all: only the gold documents a store lacks (deep rounds) -> gold_{deep,all}_docs{,_text}.npz

Queue layout: queue/ -> inprogress/ (claimed by rename, mtime = heartbeat) -> out/chunk.npz + done/. Preempted chunks rerun whole.
Document recipes reproduce the stored npz (verified cos >= 0.999): text via each model's default document path, images thumbnailed to 1024.
"""
import argparse
import glob
import importlib.util
import json
import os
import socket
import sys
import threading
import time

import numpy as np

from multi3ir import paths as _P

sys.stdout.reconfigure(line_buffering=True)
DEEP = _P.STAGE4 / "deep_nogold"
IMG_DIRS = [_P.STAGE3_SERPER / "images_deep", _P.BENCH / "corpus/images", _P.STAGE3_SERPER / "serper_images_min4",
            _P.STAGE3_SERPER / "images_textonly"]
GME_CKPT = _P.CKPTS / "gme_qwen2vl_7b_clean"


def rows(path):
    for line in open(path):
        if line.strip():
            yield json.loads(line)


def chunk_root(model, source):
    return _P.CKPTS / f"chunks_enc_{source}_{model}"


def load_image(doc_id, max_side=1024):
    """First readable copy across the image stores (corrupt files fall through), RGB, thumbnailed."""
    from PIL import Image
    for d in IMG_DIRS:
        p = d / doc_id
        if p.exists():
            try:
                im = Image.open(p); im.load()
            except Exception:
                continue
            if im.mode != "RGB":
                im = im.convert("RGB")
            if max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
            return im, str(p)
    return None, None


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def cmd_queue(a):
    emb_dir = _P.BENCH / "document_embedding" / a.model
    if a.source == "corpus":
        import pyarrow.parquet as pq
        have = set()
        text_ids = set(pq.read_table(_P.BENCH / "corpus/text_urls.parquet").column("doc_id").to_pylist())
        img_ids = set(pq.read_table(_P.BENCH / "corpus/image_urls.parquet").column("doc_id").to_pylist())
        text_files = [_P.BENCH / "corpus/text.jsonl"]
    elif a.source == "deep":
        have = {str(x) for f in ("gold_docs.npz", "gold_docs_text.npz") for x in np.load(emb_dir / f, allow_pickle=True)["ids"]}
        text_ids = {r["doc_id"] for f in ("gold_text.jsonl", "gold_text2.jsonl") for r in rows(DEEP / f)}
        img_ids = {r["doc_id"] for r in rows(DEEP / "gold_image.jsonl")}
        text_files = [DEEP / "deep_texts.jsonl"]
    else:
        have = {str(x) for f in sorted(glob.glob(str(emb_dir / "*.npz"))) for x in np.load(f, allow_pickle=True)["ids"]}
        gold = list(rows(_P.BENCH / "train/positives_gold_all.jsonl"))
        text_ids, img_ids = ({r["doc_id"] for r in gold if r["modality"] == m} for m in ("text", "image"))
        text_files = sorted(glob.glob(str(DEEP / "deep_texts*.jsonl")))
    text_ids, img_ids = text_ids - have, img_ids - have
    if a.kind == "image": text_ids = set()
    if a.kind == "text": img_ids = set()
    texts = {r["doc_id"]: r["text"] for f in (text_files if text_ids else []) for r in rows(f) if r["doc_id"] in text_ids}
    miss_t = text_ids - set(texts)
    miss_i = {i for i in img_ids if not any((d / i).exists() for d in IMG_DIRS)}
    if miss_t or miss_i:
        raise RuntimeError(f"no source for {len(miss_t)} texts / {len(miss_i)} images (e.g. {sorted(miss_t)[:2]} {sorted(miss_i)[:2]})")
    root = chunk_root(a.model, a.source)
    for sub in ("queue", "inprogress", "out", "done"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    recs_t = [{"kind": "text", "doc_id": i, "text": texts[i]} for i in sorted(text_ids)]
    recs_i = [{"kind": "image", "doc_id": i} for i in sorted(img_ids)]
    n = len(glob.glob(str(root / "*/chunk_*.jsonl")))          # continue numbering after existing chunks
    for recs in ((recs_i, recs_t) if a.image_first else (recs_t, recs_i)):
        for s in range(0, len(recs), a.chunk):
            with open(root / "queue" / f"chunk_{n:04d}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs[s:s + a.chunk])
            n += 1
    print(f"[queue] {a.model}: text {len(recs_t):,} + image {len(recs_i):,} docs -> {n} chunks under {root}/queue")


# ── encoders: recs -> (ids, fp16 L2-normalised embeddings); records whose image is unreadable are skipped ──

def _finish(ids, embs, dim):
    E = np.concatenate(embs) if embs else np.zeros((0, dim), dtype=np.float32)
    return ids, (E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)).astype(np.float16)


def _batched_images(recs, bs, run, ids, embs, item):
    """Encode image records in batches; on OOM retry one at a time and drop what still does not fit."""
    import torch
    for s in range(0, len(recs), bs):
        pairs = [(r["doc_id"], item(r["doc_id"])) for r in recs[s:s + bs]]
        pairs = [(i, x) for i, x in pairs if x is not None]
        if not pairs:
            continue
        try:
            embs.append(run([x for _, x in pairs])); ids.extend(i for i, _ in pairs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for i, x in pairs:
                try:
                    embs.append(run([x])); ids.append(i)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()


def load_mmembed():
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained("nvidia/MM-Embed", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    def enc(recs):
        ids, embs = [], []
        txt, img = [r for r in recs if r["kind"] == "text"], [r for r in recs if r["kind"] == "image"]
        with torch.no_grad():
            for s in range(0, len(txt), 8):
                b = txt[s:s + 8]
                embs.append(model.encode([{"txt": r["text"]} for r in b], max_length=2048)["hidden_states"].cpu().float().numpy())
                ids.extend(r["doc_id"] for r in b)
            _batched_images(img, 4, lambda ims: model.encode([{"txt": "", "img": im} for im in ims], max_length=4096)["hidden_states"].cpu().float().numpy(),
                            ids, embs, lambda d: load_image(d)[0])
        return _finish(ids, embs, 4096)
    return enc


def load_gme():
    import torch
    import transformers.modeling_rope_utils as ru
    ru.ROPE_INIT_FUNCTIONS.setdefault("mrope", ru.ROPE_INIT_FUNCTIONS["default"])
    if hasattr(ru, "ROPE_VALIDATION_FUNCTIONS"):
        ru.ROPE_VALIDATION_FUNCTIONS.setdefault("mrope", ru.ROPE_VALIDATION_FUNCTIONS.get("default", lambda *a, **k: None))
    gme = import_file("modeling_gme_qwen2vl", GME_CKPT / "modeling_gme_qwen2vl.py")
    model = gme.GmeQwen2VL.from_pretrained(str(GME_CKPT), torch_dtype=torch.bfloat16).cuda().eval()

    def enc(recs):
        ids, embs = [], []
        txt, img = [r for r in recs if r["kind"] == "text"], [r for r in recs if r["kind"] == "image"]
        with torch.no_grad():
            if txt:
                embs.append(model.get_text_embeddings(texts=[r["text"] for r in txt], batch_size=8).cpu().float().numpy())
                ids.extend(r["doc_id"] for r in txt)
            _batched_images(img, 4, lambda ims: model.get_image_embeddings(images=ims, batch_size=len(ims)).cpu().float().numpy(),
                            ids, embs, lambda d: load_image(d)[0])
        return _finish(ids, embs, 3584)
    return enc


def load_qwen3vl8b():
    import torch
    q = import_file("qwen3_vl_embedding", _P.QWEN_SRC / "models/qwen3_vl_embedding.py")
    emb = q.Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-8B", torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    def run(inputs):
        e = emb.process(inputs, normalize=True)
        return e.cpu().float().numpy() if torch.is_tensor(e) else e

    def enc(recs):
        ids, embs = [], []
        txt, img = [r for r in recs if r["kind"] == "text"], [r for r in recs if r["kind"] == "image"]
        with torch.no_grad():
            for s in range(0, len(txt), 8):
                b = txt[s:s + 8]
                embs.append(run([{"text": r["text"]} for r in b])); ids.extend(r["doc_id"] for r in b)
            _batched_images(img, 2, lambda paths: run([{"image": p} for p in paths]), ids, embs, lambda d: load_image(d)[1])
        return _finish(ids, embs, 4096)
    return enc


LOADERS = {"mmembed": load_mmembed, "gme_qwen2vl_7b": load_gme, "qwen3vl8b": load_qwen3vl8b}


def cmd_encode(a):
    root = a.chunk_root
    Q, IP, OUT, DONE = (os.path.join(root, d) for d in ("queue", "inprogress", "out", "done"))
    wid = f"{socket.gethostname()}.{os.getpid()}"
    enc = LOADERS[a.model]()
    print(f"[worker {wid}] {a.model} loaded")
    n = 0
    while True:
        claimed = None
        for f in sorted(os.listdir(Q)):
            if f.endswith(".jsonl"):
                try:
                    os.rename(os.path.join(Q, f), os.path.join(IP, f[:-6] + f".{wid}.jsonl")); claimed = f[:-6]; break
                except OSError:
                    continue
        if claimed is None:
            print(f"[worker {wid}] queue empty -> exit ({n} chunks)"); break
        path, outp = os.path.join(IP, f"{claimed}.{wid}.jsonl"), os.path.join(OUT, claimed + ".npz")
        if not os.path.exists(outp):
            recs = list(rows(path)); t0 = time.time()
            stop = threading.Event()

            def heartbeat():                          # keep the inprogress mtime fresh while a long chunk encodes
                while not stop.wait(120):
                    os.utime(path, None)
            threading.Thread(target=heartbeat, daemon=True).start()
            try:
                ids, E = enc(recs)
            finally:
                stop.set()
            np.savez(outp, ids=np.array(ids), embeddings=E)
            print(f"[worker {wid}] {claimed}: {len(ids)}/{len(recs)} in {time.time()-t0:.0f}s")
            n += 1
        os.rename(path, os.path.join(DONE, claimed + ".jsonl"))


def cmd_merge(a):
    roots = [chunk_root(a.model, "deep"), _P.CKPTS / f"chunks_enc_deep_{a.model}_text"] if a.source == "deep" else [chunk_root(a.model, a.source)]
    need = {r["doc_id"]: r["kind"] for root in roots for f in glob.glob(str(root / "*/chunk_*.jsonl")) for r in rows(f)}
    ids, embs = [], []
    for root in roots:
        for f in sorted(glob.glob(str(root / "out/chunk_*.npz"))):
            z = np.load(f, allow_pickle=True); ids.extend(str(x) for x in z["ids"]); embs.append(z["embeddings"])
    emb = np.concatenate(embs) if embs else np.zeros((0, 0), np.float16)
    first = {}
    for i, d in enumerate(ids):
        first.setdefault(d, i)
    missing = [d for d in need if d not in first]
    print(f"[{a.model}] queued {len(need):,}, encoded {len(first):,} unique, missing {len(missing):,}")
    if missing:
        raise SystemExit(f"not drained: e.g. {missing[:3]}")
    prefix = {"corpus": "corpus_docs", "all": "gold_all_docs", "deep": "gold_deep_docs"}[a.source]
    for kind, name in (("image", f"{prefix}.npz"), ("text", f"{prefix}_text.npz")):
        sel = [first[d] for d in need if need[d] == kind]
        out = _P.BENCH / "document_embedding" / a.model / name
        np.savez(out, ids=np.array([ids[i] for i in sel]), embeddings=emb[sel].astype(np.float16))
        print(f"[saved] {out}: {len(sel):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["queue", "encode", "merge"])
    ap.add_argument("--model", required=True, choices=list(LOADERS))
    ap.add_argument("--source", choices=["corpus", "deep", "all"], default="corpus")
    ap.add_argument("--kind", choices=["text", "image", "all"], default="all", help="queue")
    ap.add_argument("--chunk", type=int, default=500); ap.add_argument("--image-first", action="store_true")
    ap.add_argument("--chunk-root", default=None, help="encode: queue directory (default from --model/--source)")
    a = ap.parse_args()
    a.chunk_root = a.chunk_root or str(chunk_root(a.model, a.source))
    {"queue": cmd_queue, "encode": cmd_encode, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
