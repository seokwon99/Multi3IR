"""Stage 5a: gather the classifier input text for every gold document — C4 body for text, hosting-page body for images.

    python -m multi3ir.data.domain_collect text                          # gold text docs -> full_text_contents.jsonl
    python -m multi3ir.data.domain_collect links                         # gold image -> hosting-page url  (full_image_links.json)
    python -m multi3ir.data.domain_collect crawl --concurrency 64        # page bodies (trafilatura)      (full_page_texts.jsonl)
    python -m multi3ir.data.domain_collect gold-input                    # all rounds -> gold_all_input.jsonl + gold_key_map.jsonl

All outputs under results/outputs/domain_classification/; every step resumes from what is already written.
"""
import argparse
import asyncio
import glob
import hashlib
import http.client
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from multi3ir import paths as _P

D, S4, C4, SER = _P.DOMAIN, _P.STAGE4, _P.STAGE3_C4, _P.STAGE3_SERPER
UA = {"User-Agent": "Multi3IR-bot/1.0 (academic research crawler; +https://github.com/seokwon99/Multi3IR)"}


def rows(path):
    for line in open(path):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def gold_ids(modality):
    return {r["doc_id"] for r in rows(S4 / "gold_docs.jsonl") if r["modality"] == modality}


def in_shard(key, shard, num_shards):
    return num_shards == 1 or int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % num_shards == shard


def page_texts(files):
    """url -> body text from crawl outputs (first non-empty wins)."""
    out = {}
    for f in files:
        if os.path.exists(f):
            for r in rows(f):
                if r.get("text"):
                    out.setdefault(r["url"], r["text"])
    return out


def cmd_text(a):
    """C4 body of every gold text doc (doc_id = c4-<file>-<row>), one C4 file at a time, resumable per file."""
    by_file = {}
    for did in gold_ids("text"):
        _, f, row = did.split("-")
        by_file.setdefault(f, set()).add(int(row))
    out, prog = D / "full_text_contents.jsonl", D / "full_text_contents.jsonl.progress"
    done = set(prog.read_text().split()) if prog.exists() else set()
    n = 0
    with open(out, "a" if done else "w") as w, open(prog, "a") as p:
        for k, (f, want) in enumerate(sorted(by_file.items())):
            if f in done:
                continue
            for i, line in enumerate(open(C4 / "c4_pyserini_jsonl" / f"c4-train-{f}-of-01637.jsonl")):
                if i in want:
                    r = json.loads(line); w.write(json.dumps({"doc_id": r["id"], "text": r["contents"]}) + "\n"); n += 1
                if i >= max(want):
                    break
            p.write(f + "\n"); p.flush()
            if k % 50 == 0:
                print(f"  file {k}/{len(by_file)} wrote {n}", flush=True)
    print(f"[text] wrote {n} -> {out}")


def cmd_links(a):
    imgs, links = gold_ids("image"), {}
    for r in rows(SER / "serper_search_min4.jsonl"):
        for im in r["retrieved_images"]:
            if im["image_name"] in imgs:
                links.setdefault(im["image_name"], im["link"])
    json.dump(links, open(D / "full_image_links.json", "w"))
    print(f"[links] {len(links)}/{len(imgs)} images, {len(set(links.values()))} unique urls")


def cmd_crawl(a):
    import aiohttp, trafilatura
    urls = sorted(set(json.load(open(a.links or D / "full_image_links.json")).values()))
    urls = [u for u in urls if in_shard(u, a.shard, a.num_shards)]
    out = a.out or (D / "full_page_texts.jsonl" if a.num_shards == 1 else D / f"deep_page_texts_{a.shard}of{a.num_shards}.jsonl")
    done = {r["url"] for r in rows(out) if r.get("text") or not a.retry} if os.path.exists(out) else set()
    todo = [u for u in urls if u not in done]
    print(f"[crawl] {len(urls)} urls, {len(done)} done, {len(todo)} to crawl", flush=True)
    sem = asyncio.Semaphore(a.concurrency)

    async def fetch(session, url):
        async with sem:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20), headers=UA, allow_redirects=True) as r:
                    if r.status != 200 or "html" not in r.headers.get("content-type", ""):
                        return url, None
                    html = await r.text(errors="ignore")
                txt = trafilatura.extract(html, include_comments=False)
            except Exception:
                return url, None
            return url, (txt if txt and len(txt.split()) >= 30 else None)

    async def run():
        ok, t0 = 0, time.time()
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=a.conn_limit, ssl=False)) as s, open(out, "a") as w:
            for c in range(0, len(todo), 2000):
                for fut in asyncio.as_completed([fetch(s, u) for u in todo[c:c + 2000]]):
                    u, txt = await fut
                    w.write(json.dumps({"url": u, "text": txt}) + "\n"); ok += bool(txt)
                w.flush()
                print(f"  {min(c + 2000, len(todo))}/{len(todo)} ok={ok} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[crawl] done ok={ok}")
    asyncio.run(run())


def cmd_gold_input(a):
    """Classifier input for every gold doc of every round: {key, text, kind}; key = doc_id (text), page url (image with page)
    or image_name (image with title only). gold_key_map.jsonl maps each gold pair to its key."""
    files = [S4 / "gold_docs.jsonl"] + [S4 / "deep_nogold" / n for n in
                                        ("gold_text.jsonl", "gold_image.jsonl", "gold_text2.jsonl", "gold_image_textonly.jsonl", "gold_text3.jsonl")
                                        if (S4 / "deep_nogold" / n).exists()]
    gold = [(r["uid"], r["perspective_idx"], r["doc_id"], r["modality"], f.name) for f in files for r in rows(f)]
    txt_ids, img_ids = ({d for _, _, d, m, _ in gold if m == mod} for mod in ("text", "image"))
    text = {r["doc_id"]: r["text"] for r in rows(D / "full_text_contents.jsonl") if r["doc_id"] in txt_ids}
    for f in (C4 / n for n in ("c4_deep_rank11_20.jsonl", "c4_deep_dense10_30_all.jsonl", "c4_imageonly_dense10_30_all.jsonl")):
        if f.exists():
            for r in rows(f):
                for d in r["retrieved_documents"]:
                    if d["doc_id"] in txt_ids:
                        text.setdefault(d["doc_id"], d["text"])
    meta = {}
    for f in (SER / n for n in ("serper_search_min4.jsonl", "image_deep_nogold.jsonl", "image_textonly_page1.jsonl", "image_textonly_page2.jsonl")):
        for r in rows(f):
            for im in r.get("retrieved_images") or []:
                if im.get("image_name") in img_ids:
                    meta.setdefault(im["image_name"], im)
    page = page_texts([D / "full_page_texts.jsonl"] + sorted(glob.glob(str(D / "deep_page_texts_*.jsonl"))))
    print(f"gold pairs {len(gold):,}: text {len(text):,}/{len(txt_ids):,}, image meta {len(meta):,}/{len(img_ids):,}", flush=True)
    key_of, out = {}, {}
    for n, im in meta.items():
        u = im.get("link")
        if u and u in page:
            key_of[n] = u; out[u] = (page[u], "image_page")
        else:
            t = " ".join(x for x in (im.get("title", ""), im.get("source", ""), (u or "").split("//")[-1].split("/")[0]) if x).strip()
            if t:
                key_of[n] = n; out[n] = (t, "image_title")
    for d, t in text.items():
        key_of[d] = d; out[d] = (t, "text")
    with open(D / "gold_all_input.jsonl", "w") as w:
        for k, (t, kind) in out.items():
            w.write(json.dumps({"key": k, "text": t, "kind": kind}, ensure_ascii=False) + "\n")
    miss = Counter()
    with open(D / "gold_key_map.jsonl", "w") as w:
        for u, p, d, m, src in gold:
            miss[src] += key_of.get(d) is None
            w.write(json.dumps({"uid": u, "perspective_idx": p, "doc_id": d, "modality": m, "src": src, "key": key_of.get(d)}) + "\n")
    print(f"[gold-input] {len(out):,} inputs {dict(Counter(v[1] for v in out.values()))}; pairs without text {dict(miss)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["text", "links", "crawl", "gold-input"])
    ap.add_argument("--links", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--retry", action="store_true", help="crawl: re-crawl urls recorded with no text")
    ap.add_argument("--concurrency", type=int, default=64); ap.add_argument("--conn-limit", type=int, default=128)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    {"text": cmd_text, "links": cmd_links, "crawl": cmd_crawl, "gold-input": cmd_gold_input}[a.cmd](a)


if __name__ == "__main__":
    main()
