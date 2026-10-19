"""Stage 3b: image evidence — Google Images via Serper (10 per perspective), then bulk download.

    python -m multi3ir.data.stage3_image search                                           # -> serper_search_min4.jsonl (resumable)
    python -m multi3ir.data.stage3_image download --inputs <search.jsonl> --img-dir <dir> --shard 0 --num-shards 8
    python -m multi3ir.data.stage3_image deep-search --scope nogold --page 2                # deep round: page 2 for gold-less perspectives

Key: $SERPER_API_KEY or <ROOT>/serper_api.txt. Image file name = serper_<sha256(url)[:12]>.jpg.
"""
import argparse
import hashlib
import http.client
import io
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

from multi3ir import paths as _P

SEARCH_OUT = _P.STAGE3_SERPER / "serper_search_min4.jsonl"
IMG_DIR = _P.STAGE3_SERPER / "serper_images_min4"
DEEP_DIR = _P.STAGE4 / "deep_nogold"
UA = "Multi3IR-bot/1.0 (academic research crawler; +https://github.com/seokwon99/Multi3IR)"


def api_key():
    key = os.environ.get("SERPER_API_KEY") or ((_P.ROOT / "serper_api.txt").read_text().strip()
                                               if (_P.ROOT / "serper_api.txt").exists() else "")
    if not key:
        raise RuntimeError(f"no Serper key: set SERPER_API_KEY or write it to {_P.ROOT / 'serper_api.txt'}")
    return key


def img_name(url):
    return f"serper_{hashlib.sha256(url.encode()).hexdigest()[:12]}.jpg"


def serper_search(key, query, num=10, page=1):
    """Returns (images, credits, error). Any non-200 or a body without "images" (e.g. "Not enough credits") is an error,
    never an empty result, so a failed query is retried on the next run instead of being marked done."""
    c = http.client.HTTPSConnection("google.serper.dev", timeout=30)
    try:
        c.request("POST", "/images", json.dumps({"q": query, "num": num, "page": page}),
                  {"X-API-KEY": key, "Content-Type": "application/json"})
        resp = c.getresponse(); body = resp.read().decode(); d = json.loads(body)
        if resp.status != 200 or "images" not in d:
            return [], 0, f"http_{resp.status}:{d.get('message', body[:80])}"
        return d["images"], d.get("credits", 0), None
    except Exception as e:
        return [], 0, type(e).__name__
    finally:
        c.close()


def image_rows(images, first_rank):
    return [{"rank": first_rank + r, "image_name": img_name(x["imageUrl"]), "image_url": x["imageUrl"],
             "title": x.get("title", ""), "source": x.get("source", ""), "domain": x.get("domain", ""),
             "link": x.get("link", "")} for r, x in enumerate(images) if x.get("imageUrl")]


def done_keys(*paths):
    done = set()
    for p in paths:
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line); done.add((r["uid"], r["perspective_idx"]))
    return done


def run_searches(todo, out, workers, page, method=None):
    """todo: [(uid, pidx, question, summary)]; appends one row per perspective; aborts on credit exhaustion."""
    key = api_key(); credits = 0; errs = Counter(); t0 = time.time()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a") as w, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(serper_search, key, s, 10, page): (u, i, q, s) for u, i, q, s in todo}
        for k, fu in enumerate(as_completed(futs)):
            u, i, q, s = futs[fu]
            imgs, cr, err = fu.result(); credits += cr
            if err:
                errs[err] += 1
                if "Not enough credits" in err:
                    print(f"\n[abort] {err} at {k+1:,}", flush=True)
                    for f2 in futs: f2.cancel()
                    break
                continue
            row = {"uid": u, "perspective_idx": i, "question": q, "summary": s, "retrieved_images": image_rows(imgs, (page - 1) * 10)}
            if method:
                row["method"] = method
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (k + 1) % 5000 == 0:
                w.flush(); el = time.time() - t0
                print(f"  {k+1:,}/{len(todo):,} credits {credits:,} {(k+1)/el:.0f}/s ETA {(len(todo)-k-1)/((k+1)/el)/60:.0f}min", flush=True)
    print(f"[credits] {credits:,} used; errors {dict(errs)} (rerun retries them)\n[done] -> {out}", flush=True)


def cmd_search(a):
    todo = []
    for line in open(_P.PERSPECTIVES):
        d = json.loads(line)
        q = d.get("Question") or d.get("question") or d.get("Title", "")
        for i, p in enumerate(d.get("perspectives") or []):
            s = p.get("summary", "") if isinstance(p, dict) else (p[0] if isinstance(p, (list, tuple)) else p)
            if s:
                todo.append((d["uid"], i, q, s))
    done = done_keys(a.out)
    todo = [x for x in todo if (x[0], x[1]) not in done][:a.limit or None]
    print(f"[search] {len(todo):,} perspectives to search ({len(done):,} done)", flush=True)
    run_searches(todo, a.out, a.workers, 1)


def cmd_deep_search(a):
    """Buy page N for perspectives with no gold (nogold) or text-only gold (textonly), from queries.jsonl positives."""
    mods = {}
    for name in ("gold_docs.jsonl", "deep_nogold/gold_text.jsonl", "deep_nogold/gold_image.jsonl", "deep_nogold/gold_text2.jsonl"):
        for line in open(_P.STAGE4 / name):
            d = json.loads(line); mods.setdefault((d["uid"], d["perspective_idx"]), set()).add(d["modality"])
    want = (lambda k: k not in mods) if a.scope == "nogold" else (lambda k: mods.get(k) == {"text"})
    todo = [(d["uid"], i, d.get("query", ""), p) for d in map(json.loads, open(_P.TRAIN_QUERIES))
            for i, p in enumerate(d.get("positives") or []) if want((d["uid"], i))]
    out = a.out or str(_P.STAGE3_SERPER / (f"image_deep_nogold.jsonl" if a.scope == "nogold" else f"image_textonly_page{a.page}.jsonl"))
    done = done_keys(out)
    todo = [x for x in todo if (x[0], x[1]) not in done][:a.limit or None]
    print(f"[{a.scope}] {len(todo):,} to search ({len(done):,} done), ~{len(todo)} credits", flush=True)
    run_searches(todo, out, a.workers, a.page, method=f"page{a.page}")


def fetch(name, url, img_dir, max_side, referer, retries=1, timeout=15):
    """One GET with an identified research user agent; one retry on 429/5xx; any PIL-readable format -> JPEG."""
    if url.startswith("//"):
        url = "https:" + url
    elif not url.startswith(("http://", "https://")):
        url = "https://" + url.lstrip("/")
    if url.startswith("x-raw-image:"):
        return name, False, "x-raw-image"
    headers = {"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"}
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code in (429, 500, 502, 503, 202) and attempt < retries:
                time.sleep(2 + attempt * 3); continue
            if r.status_code != 200:
                return name, False, f"http_{r.status_code}"
            ctype = r.headers.get("content-type", "").split(";")[0].strip()
            if ctype == "image/svg+xml" or b"<svg" in r.content[:1000]:
                try:
                    import cairosvg
                    im = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=r.content, output_width=1024)))
                except Exception:
                    return name, False, "svg_unrendered"
            elif ctype.startswith("text/html"):
                return name, False, "html_not_image"
            else:
                im = Image.open(io.BytesIO(r.content))
            im.load()
            if im.mode in ("RGBA", "P", "LA"):
                im = im.convert("RGB")
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
            im.save(os.path.join(img_dir, name), "JPEG", quality=92)
            return name, True, "ok"
        except requests.exceptions.Timeout:
            if attempt == retries:
                return name, False, "timeout"
        except Exception as e:
            return name, False, type(e).__name__
    return name, False, "retries_exhausted"


def cmd_download(a):
    """Unique images across --inputs not yet in --img-dir; --num-shards splits by name hash (network-bound, run on many nodes)."""
    os.makedirs(a.img_dir, exist_ok=True)
    have, seen, todo = set(os.listdir(a.img_dir)), set(), []
    for path in a.inputs:
        for line in open(path):
            for x in json.loads(line)["retrieved_images"]:
                n, u = x["image_name"], x["image_url"]
                if not u or n in seen or n in have:
                    continue
                seen.add(n)
                if a.num_shards > 1 and int(hashlib.sha256(n.encode()).hexdigest()[:8], 16) % a.num_shards != a.shard:
                    continue
                todo.append((n, u, x.get("link")))
    print(f"[download] {len(todo):,} images (shard {a.shard}/{a.num_shards})", flush=True)
    ok, reasons, t0 = 0, Counter(), time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(fetch, n, u, a.img_dir, a.max_side, ref) for n, u, ref in todo]
        for i, fu in enumerate(as_completed(futs)):
            _, good, why = fu.result(); ok += good; reasons[why] += 1
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  {i+1:,}/{len(todo):,} ok {ok:,} {(i+1)/el:.0f}/s ETA {(len(todo)-i-1)/((i+1)/el)/60:.0f}min", flush=True)
    print(f"[download] ok {ok:,}/{len(todo):,}; reasons {dict(reasons.most_common(8))}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["search", "deep-search", "download"])
    ap.add_argument("--workers", type=int, default=None, help="search 20 / download 48")
    ap.add_argument("--limit", type=int, default=0, help="search: only the first N pending perspectives (0 = all)")
    ap.add_argument("--out", default=None, help="search output jsonl")
    ap.add_argument("--scope", default="nogold", choices=["nogold", "textonly"], help="deep-search")
    ap.add_argument("--page", type=int, default=2, help="deep-search: Serper page (10 images each)")
    ap.add_argument("--inputs", nargs="+", default=[str(SEARCH_OUT)], help="download: search jsonl files")
    ap.add_argument("--img-dir", default=str(IMG_DIR))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-side", type=int, default=0, help="download: cap longest edge (0 = original)")
    a = ap.parse_args()
    a.workers = a.workers or (48 if a.cmd == "download" else 20)
    if a.cmd == "search":
        a.out = a.out or str(SEARCH_OUT)
    {"search": cmd_search, "deep-search": cmd_deep_search, "download": cmd_download}[a.cmd](a)


if __name__ == "__main__":
    main()
