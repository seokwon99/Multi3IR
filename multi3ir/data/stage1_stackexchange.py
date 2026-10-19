"""Stage 1: download StackExchange dumps (archive.org), extract Posts.xml, keep well-answered questions.

    python -m multi3ir.data.stage1_stackexchange --base_dir results/outputs/stage1_raw_data/stackexchange_filtered

Output: <base_dir>/<category>/<site>/<site>_filtered.jsonl with {Title, Body, Answers, Links}, plus filter_stats.json.
"""
import argparse
import html
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

SITES = Path(__file__).with_name("stage1_sites.json")   # {category: {site: {site_url, archive_url}}}
ARCHIVE = "https://archive.org/download/stackexchange/"


def safe_name(s):
    s = re.sub(r"[\\/:\*\?\"<>\|\n\r\t]+", "_", (s or "").strip().replace("&", "_"))
    return re.sub(r"\s+", "_", s) or "unnamed"


def url_exists(url):
    return subprocess.run(["wget", "--spider", "-q", url], capture_output=True).returncode == 0


def resolve_archive_url(site_url):
    """{domain}.7z, else {domain}-Posts.7z (split dumps), else search the archive.org file list."""
    domain = (urlparse(site_url).netloc if "://" in site_url else site_url).strip("/").replace("www.", "")
    for cand in (f"{ARCHIVE}{domain}.7z", f"{ARCHIVE}{domain}-Posts.7z"):
        if url_exists(cand):
            return cand
    import requests
    files = [f["name"] for f in requests.get("https://archive.org/metadata/stackexchange", timeout=60).json()["files"]]
    cands = sorted((fn for fn in files if fn.startswith(domain) and fn.endswith(".7z")), key=len)
    if not cands:
        raise FileNotFoundError(f"no .7z dump for {domain}")
    return ARCHIVE + cands[0]


def download_and_extract(archive_url, target_dir, tries):
    """Ensure <target_dir>/Posts.xml exists; returns its path."""
    posts = os.path.join(target_dir, "Posts.xml")
    if os.path.exists(posts):
        return posts
    os.makedirs(target_dir, exist_ok=True)
    archive = os.path.join(target_dir, os.path.basename(urlparse(archive_url).path))
    if not os.path.exists(archive):
        subprocess.run(["wget", "-c", "--content-disposition", "--trust-server-names", "--retry-connrefused",
                        f"--tries={tries}", "--waitretry=5", "-O", archive, archive_url], check=True)
    import py7zr
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=target_dir)
    found = next((os.path.join(r, f) for r, _, fs in os.walk(target_dir) for f in fs if f.lower() == "posts.xml"), None)
    if found is None:
        raise FileNotFoundError(f"Posts.xml not found under {target_dir}")
    if os.path.abspath(found) != os.path.abspath(posts):
        shutil.copy2(found, posts)
    return posts


def filter_posts(posts_xml, out_jsonl, min_answer_count, min_answer_score, min_good_answers, min_links):
    """Questions with >= min_answer_count answers, >= min_good_answers answers scoring >= min_answer_score,
    and >= min_links hyperlinks across those answers. Returns (raw_questions, kept)."""
    raw, questions, answers, pending = 0, {}, {}, {}
    for _, e in ET.iterparse(posts_xml, events=("end",)):
        if e.tag != "row":
            continue
        if e.get("PostTypeId") == "1":
            raw += 1
            qid, n_ans = e.get("Id"), int(e.get("AnswerCount") or 0)
            if qid and n_ans >= min_answer_count:
                questions[qid] = {"Title": e.get("Title") or "", "Body": e.get("Body") or ""}
                if qid in pending:
                    answers[qid] = pending.pop(qid)
        elif e.get("PostTypeId") == "2":
            pid = e.get("ParentId")
            if pid and int(e.get("Score") or 0) >= min_answer_score:
                (answers if pid in questions else pending).setdefault(pid, []).append(e.get("Body") or "")
        e.clear()
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for qid, q in questions.items():
            ans = answers.get(qid, [])
            if len(ans) < min_good_answers:
                continue
            links = [a["href"] for body in ans for a in BeautifulSoup(body, "html.parser").find_all("a", href=True)]
            if len(links) < min_links:
                continue
            f.write(json.dumps({"Title": q["Title"], "Body": html.unescape(q["Body"]),
                                "Answers": [html.unescape(a) for a in ans], "Links": links}, ensure_ascii=False) + "\n")
            kept += 1
    return raw, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(SITES), help="category -> site -> {site_url, archive_url} json")
    ap.add_argument("--base_dir", required=True, help="download/extract root; filtered jsonl goes beside each Posts.xml")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--min_answer_count", type=int, default=3)
    ap.add_argument("--min_answer_score", type=int, default=3)
    ap.add_argument("--min_good_answers", type=int, default=1)
    ap.add_argument("--min_links", type=int, default=2)
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    stats = {"filter": {k: getattr(args, k) for k in ("min_answer_count", "min_answer_score", "min_good_answers", "min_links")},
             "categories": {}}
    for category, sites in tqdm(spec.items(), desc="categories"):
        cat = safe_name(category)
        for site, entry in tqdm(sites.items(), desc=cat, leave=False):
            link = safe_name(site)
            site_url = entry["site_url"] if isinstance(entry, dict) else entry
            try:
                archive_url = (entry.get("archive_url") if isinstance(entry, dict) else None) or resolve_archive_url(site_url)
                target = os.path.join(args.base_dir, cat, link)
                posts = download_and_extract(archive_url, target, args.tries)
                raw, kept = filter_posts(posts, os.path.join(target, f"{link}_filtered.jsonl"), args.min_answer_count,
                                         args.min_answer_score, args.min_good_answers, args.min_links)
                stats["categories"].setdefault(cat, {})[link] = {"site_url": site_url, "archive_url": archive_url,
                                                                 "raw_questions": raw, "filtered_questions": kept}
                print(f"[{cat}/{link}] {kept}/{raw} questions kept", flush=True)
            except Exception as e:
                print(f"[ERROR] {category}/{site}: {e}", flush=True)
                stats["categories"].setdefault(cat, {})[link] = {"site_url": site_url, "error": str(e)}
    out = os.path.join(args.base_dir, "filter_stats.json")
    json.dump(stats, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[stats] -> {out}")


if __name__ == "__main__":
    main()
