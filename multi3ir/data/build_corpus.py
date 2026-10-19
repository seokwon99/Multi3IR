"""Materialise results/bench/corpus/ (images/ + text.jsonl): doc ids + source urls come from corpus/{text,image}_urls.parquet
(the evaluation corpus, written by `hub pull`) and from the gold positives files.

    python -m multi3ir.data.build_corpus --check                          # what is already present
    python -m multi3ir.data.build_corpus --images --workers 64            # download image documents from doc_url
    python -m multi3ir.data.build_corpus --text --c4-dir <c4_pyserini_jsonl>   # C4 passages by doc_id (eval passages come inline)

Both builders skip documents already on disk, so an interrupted run resumes with the same command.
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from multi3ir import paths as _P

CORPUS = _P.BENCH / "corpus"
IMG_DIR, TEXT_JSONL = CORPUS / "images", CORPUS / "text.jsonl"


def rows(path):
    for line in open(path):
        if line.strip():
            yield json.loads(line)


def wanted_docs():
    """(image doc_id -> url, text doc_ids) over corpus/*_urls.parquet, positives_gold (flat) and positives_specific (nested)."""
    images, texts = {}, set()
    if (CORPUS / "image_urls.parquet").exists():
        import pyarrow.parquet as pq
        images.update(zip(*(pq.read_table(CORPUS / "image_urls.parquet").column(c).to_pylist() for c in ("doc_id", "url"))))
        texts |= set(pq.read_table(CORPUS / "text_urls.parquet").column("doc_id").to_pylist())
    for path in (_P.TRAIN_POSITIVES, _P.EVAL_GOLD):
        if not path.exists():
            print(f"[skip] {path} not found"); continue
        for r in rows(path):
            if "modality" in r:
                if r["modality"] == "image" and r.get("doc_url"):
                    images.setdefault(r["doc_id"], r["doc_url"])
                elif r["modality"] == "text":
                    texts.add(r["doc_id"])
            else:
                texts |= {d["doc_id"] for d in r.get("retrieved_documents") or []}
                for im in r.get("retrieved_images") or []:
                    if im.get("image_url"):
                        images.setdefault(im["image_name"], im["image_url"])
    return images, texts


def present_text_ids():
    return {r["doc_id"] for r in rows(TEXT_JSONL)} if TEXT_JSONL.exists() else set()


def build_images(images, workers, timeout):
    from io import BytesIO
    import requests
    from PIL import Image
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(d, u) for d, u in images.items() if not (IMG_DIR / d).exists()]
    print(f"[images] {len(todo):,} to download", flush=True)

    def fetch(item):
        """Decode + re-encode like stage 3 (rejects HTML-with-200 and truncated bytes); write via .part so a kill leaves no partial file."""
        doc_id, url = item
        dst, tmp = IMG_DIR / doc_id, IMG_DIR / (doc_id + ".part")
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Multi3IR-bot/1.0 (academic research crawler; +https://github.com/seokwon99/Multi3IR)"})
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)); img.load()
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img.save(tmp, quality=95); tmp.rename(dst)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            return False
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, good in enumerate(pool.map(fetch, todo), 1):
            ok += good
            if i % 2000 == 0:
                print(f"  {i:,}/{len(todo):,} ok={ok:,}", flush=True)
    print(f"[images] ok={ok:,} fail={len(todo)-ok:,} -> {IMG_DIR} (source urls rot; rerun to retry the missing ones)")


def build_text(texts, c4_dir):
    todo = texts - present_text_ids()
    print(f"[text] {len(todo):,} to extract", flush=True)
    CORPUS.mkdir(parents=True, exist_ok=True)
    found = 0
    with open(TEXT_JSONL, "a") as out:
        if _P.EVAL_GOLD.exists():                     # eval passages ship inline
            for r in rows(_P.EVAL_GOLD):
                for d in r.get("retrieved_documents") or []:
                    if d["doc_id"] in todo and d.get("text"):
                        out.write(json.dumps({"doc_id": d["doc_id"], "url": d.get("url", ""), "text": d["text"]}, ensure_ascii=False) + "\n")
                        todo.discard(d["doc_id"]); found += 1
        if todo and not c4_dir:
            print(f"  {len(todo):,} C4 passages still missing: pass --c4-dir (Pyserini JSONL copy of C4)")
        elif todo:
            by_shard = defaultdict(set)
            for doc_id in todo:                        # c4-<shard>-<row>; matched on the id field, not the line number
                by_shard[doc_id.split("-")[1]].add(doc_id)
            for shard, want in sorted(by_shard.items()):
                path = os.path.join(c4_dir, f"c4-train-{shard}-of-01637.jsonl")
                if not os.path.exists(path):
                    print(f"  [miss] {path}"); continue
                for d in rows(path):
                    if d.get("id") in want:
                        out.write(json.dumps({"doc_id": d["id"], "url": d.get("url", ""), "text": d.get("contents", "")}, ensure_ascii=False) + "\n")
                        found += 1
    print(f"[text] {found:,} passages -> {TEXT_JSONL}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--images", action="store_true"); ap.add_argument("--text", action="store_true")
    ap.add_argument("--c4-dir", default=None); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--timeout", type=int, default=20)
    a = ap.parse_args()
    if not (a.check or a.images or a.text):
        ap.error("pass at least one of --check / --images / --text")
    images, texts = wanted_docs()
    if not images and not texts:
        raise SystemExit(f"no documents found: run `python -m multi3ir.data.hub pull` first ({_P.BENCH})")
    print(f"images {sum((IMG_DIR / d).exists() for d in images):,}/{len(images):,} present   {IMG_DIR}")
    print(f"text   {len(present_text_ids() & texts):,}/{len(texts):,} present   {TEXT_JSONL}")
    if a.images:
        build_images(images, a.workers, a.timeout)
    if a.text:
        build_text(texts, a.c4_dir)


if __name__ == "__main__":
    main()
