"""Stage 4b: gold = stage-1 "fully" AND no sibling judged "fully" in stage 2; then the released positives tables.

    python -m multi3ir.data.stage4_gold merge                                  # rejudge stage1/2 (+unknown fixes) -> gold_docs.jsonl
    python -m multi3ir.data.stage4_gold round --stage1 '<glob>' --stage2 '<glob>' --stage2-input <jsonl> \
        --modality image --round <name> --out deep_nogold/gold_image.jsonl      # one deep retrieval round
    python -m multi3ir.data.stage4_gold positives                              # gold_docs -> bench/train/positives_gold.jsonl (WebOrganizer domains)
    python -m multi3ir.data.stage4_gold positives-all                          # every round -> positives_gold_all.jsonl (nvidia domains)

Row schema: {uid, perspective_idx, doc_id, modality, [round], domain, domain_source, local_doc_path, doc_url}.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

from multi3ir import paths as _P

S4, C4, SER, DOM = _P.STAGE4, _P.STAGE3_C4, _P.STAGE3_SERPER, _P.DOMAIN
GOLD = S4 / "gold_docs.jsonl"
DEEP = S4 / "deep_nogold"
ROUNDS = [("rank0_9", GOLD), ("deep_text_bm25_11_20", DEEP / "gold_text.jsonl"), ("deep_image_page2", DEEP / "gold_image.jsonl"),
          ("dense_11_30_nogold", DEEP / "gold_text2.jsonl"), ("image_textonly_page1_2", DEEP / "gold_image_textonly.jsonl"),
          ("dense_11_30_imageonly", DEEP / "gold_text3.jsonl")]
IMG_DIRS = [SER / "serper_images_min4", SER / "images_deep", SER / "images_textonly", _P.BENCH / "corpus/images"]


def rows(path):
    for line in open(path):
        if line.strip():
            yield json.loads(line)


def load_topics(pattern):
    """key -> topic from classifier outputs; malformed lines are counted and skipped."""
    out, bad = {}, 0
    for path in sorted(glob.glob(str(pattern))):
        for line in open(path):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1; continue
            if d.get("topic"):
                out[d["key"]] = d["topic"]
    if bad:
        print(f"[topics] {pattern}: {bad} malformed lines skipped")
    return out


def text_urls(pools, needed):
    """doc_id -> url from retrieval pools (first occurrence wins)."""
    out = {}
    for p in pools:
        if os.path.exists(p):
            for r in rows(p):
                for d in r.get("retrieved_documents") or []:
                    if d.get("doc_id") in needed and d.get("url"):
                        out.setdefault(d["doc_id"], d["url"])
    return out


def image_meta(pools, needed):
    """image_name -> (link, image_url) from search files (first occurrence wins)."""
    out = {}
    for p in pools:
        if os.path.exists(p):
            for r in rows(p):
                for d in r.get("retrieved_images") or []:
                    if d.get("image_name") in needed:
                        out.setdefault(d["image_name"], (d.get("link", ""), d.get("image_url", "")))
    return out


def write_positives(out, records, stats):
    with open(str(out) + ".tmp", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(str(out) + ".tmp", out)
    print(f"[done] {out}: rows={len(records):,} uids={len({r['uid'] for r in records}):,} "
          f"perspectives={len({(r['uid'], r['perspective_idx']) for r in records}):,}")
    for k, v in stats.items():
        print(f"  {k}: {dict(v) if isinstance(v, Counter) else v}")


def cmd_merge(a):
    """Rejudged stage 1 + chunked stage 2 (unknown re-judge fixes applied) -> gold_docs.jsonl, rejudge_stage2_FINAL.jsonl, gold_summary.txt."""
    s1 = {(r["uid"], r["perspective_idx"], r["doc_id"], r["modality"]): r["support"] for r in rows(S4 / "rejudge_stage1_FINAL.jsonl")}
    if (S4 / "rejudge_stage1_unknowns_fixed.jsonl").exists():
        for r in rows(S4 / "rejudge_stage1_unknowns_fixed.jsonl"):
            k = (r["uid"], r["perspective_idx"], r["doc_id"], r["modality"])
            if s1.get(k) == "unknown":
                s1[k] = r["support"]
    fully = {k for k, v in s1.items() if v == "fully"}
    fix2 = {(r["uid"], r["perspective_idx"], r["doc_id"], r["modality"], r["sibling_perspective_idx"]): r
            for r in rows(S4 / "rejudge_stage2_unknowns_fixed.jsonl")} if (S4 / "rejudge_stage2_unknowns_fixed.jsonl").exists() else {}
    sib_fully, n2 = set(), 0
    with open(S4 / "rejudge_stage2_FINAL.jsonl", "w") as fo:
        for f in glob.glob(str(S4 / "chunks_s2/out/chunk_*.jsonl")):
            for r in rows(f):
                k5 = (r["uid"], r["perspective_idx"], r["doc_id"], r["modality"], r["sibling_perspective_idx"])
                if r["sibling_support"] == "unknown" and k5 in fix2:
                    r = fix2[k5]
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n2 += 1
                if r["sibling_support"] == "fully":
                    sib_fully.add(k5[:4])
    gold = [k for k in fully if k not in sib_fully]
    with open(GOLD, "w") as fo:
        for uid, pidx, doc, mod in gold:
            fo.write(json.dumps({"uid": uid, "perspective_idx": pidx, "doc_id": doc, "modality": mod}, ensure_ascii=False) + "\n")
    cm = Counter(k[3] for k in gold)
    summary = (f"stage1 fully docs: {len(fully):,}\ndropped (fully-for-sibling): {len(fully)-len(gold):,}\n"
               f"GOLD (exclusive): {len(gold):,} = text {cm['text']:,} + image {cm['image']:,}\n"
               f"GOLD perspectives: {len({k[:2] for k in gold}):,}\nstage2 verdicts: {n2:,}\n")
    print(summary); open(S4 / "gold_summary.txt", "w").write(summary)


def cmd_round(a):
    """Gold for one retrieval round; refuses to write if a fully doc still lacks sibling verdicts (stage 2 unfinished)."""
    s1 = {(r["uid"], r["perspective_idx"], r["doc_id"]): r["support"] for f in sorted(glob.glob(a.stage1)) for r in rows(f)}
    fully = [k for k, v in s1.items() if v == "fully"]
    sib = defaultdict(list)
    for f in sorted(glob.glob(a.stage2)):
        for r in rows(f):
            sib[(r["uid"], r["perspective_idx"], r["doc_id"])].append(r["sibling_support"])
    n_persp = Counter(r["uid"] for r in rows(a.stage2_input))
    missing = [k for k in fully if len(sib.get(k, [])) < n_persp.get(k[0], 1) - 1]
    print(f"[{a.round}] stage1 {len(s1):,} pairs, fully {len(fully):,}; fully docs missing sibling verdicts: {len(missing):,}")
    if missing and not a.allow_missing:
        raise SystemExit("stage 2 incomplete — not writing (use --allow-missing to override)")
    gold = [k for k in fully if "fully" not in sib.get(k, [])]
    with open(a.out, "w") as w:
        for uid, pidx, did in gold:
            w.write(json.dumps({"uid": uid, "perspective_idx": pidx, "doc_id": did, "modality": a.modality, "round": a.round}) + "\n")
    print(f"[{a.round}] gold {len(gold):,} docs over {len({k[:2] for k in gold}):,} perspectives -> {a.out}")


def cmd_positives(a):
    """positives_gold.jsonl: round-0 gold with WebOrganizer subject domains; joins other than image domain must be total."""
    gold = list(rows(GOLD))
    urls = text_urls(sorted(glob.glob(str(C4 / "c4_dense_reranked_top10_min4_shard*.jsonl"))), {r["doc_id"] for r in gold if r["modality"] == "text"})
    meta = image_meta([SER / "serper_search_min4.jsonl"], {r["doc_id"] for r in gold if r["modality"] == "image"})
    t_text, t_page, t_title = (load_topics(DOM / f) for f in ("full_topics_text.jsonl", "full_topics_image.jsonl", "full_topics_image_titleurl.jsonl"))
    local = set(os.listdir(IMG_DIRS[0]))
    eval_uids = {u.strip() for u in open(_P.HELDOUT_UIDS) if u.strip()}
    out, stats = [], {"by_modality": Counter(), "domain_source": Counter(), "domain_null": Counter()}
    for r in gold:
        uid, pidx, did, mod = r["uid"], r["perspective_idx"], r["doc_id"], r["modality"]
        if uid in eval_uids:
            raise RuntimeError(f"eval uid in gold_docs: {uid}")
        if mod == "text":
            url, path, domain, src = urls[did], None, t_text.get(did), "page_text"
        else:
            if did not in meta or did not in local:
                raise RuntimeError(f"image {did} missing in serper meta or {IMG_DIRS[0]}")
            link, url = meta[did]
            path = str(IMG_DIRS[0].relative_to(_P.ROOT) / did)
            domain, src = (t_page[link], "page_text") if link in t_page else (t_title.get(did), "title_url")
        src = src if domain is not None else None
        out.append({"uid": uid, "perspective_idx": pidx, "doc_id": did, "modality": mod, "domain": domain,
                    "domain_source": src, "local_doc_path": path, "doc_url": url})
        stats["by_modality"][mod] += 1; stats["domain_source"][src] += 1; stats["domain_null"][mod] += domain is None
    write_positives(_P.TRAIN_POSITIVES, out, stats)


def cmd_positives_all(a):
    """positives_gold_all.jsonl: every round, nvidia multilingual domains via gold_key_map; misses are counted, not raised."""
    gold = [dict(r, round=name) for name, path in ROUNDS for r in rows(path)]
    need_t, need_i = ({r["doc_id"] for r in gold if r["modality"] == m} for m in ("text", "image"))
    urls = text_urls(sorted(glob.glob(str(C4 / "c4_dense_reranked_top10_min4_shard*.jsonl"))) +
                     [C4 / n for n in ("c4_deep_rank11_20.jsonl", "c4_deep_dense10_30_all.jsonl", "c4_imageonly_dense10_30_all.jsonl")], need_t)
    meta = image_meta([SER / n for n in ("serper_search_min4.jsonl", "image_deep_nogold.jsonl", "image_textonly_page1.jsonl",
                                          "image_textonly_page2.jsonl")], need_i)
    where = {}
    for d in IMG_DIRS:
        if d.is_dir():
            for n in os.listdir(d):
                if n in need_i:
                    where.setdefault(n, str(d.relative_to(_P.ROOT) / n))
    topic = load_topics(DOM / "full_topics_gold_nvidia_ml_*of*.jsonl")
    key_of = {(r["uid"], r["perspective_idx"], r["doc_id"]): r["key"] for r in rows(DOM / "gold_key_map.jsonl")}
    print(f"[load] text urls {len(urls):,}/{len(need_t):,}; image meta {len(meta):,}/{len(need_i):,}; files {len(where):,}; topics {len(topic):,}")
    out, stats = [], {"by_modality": Counter(), "by_round": Counter(), "domain_source": Counter(), "missing": Counter()}
    for r in gold:
        uid, pidx, did, mod = r["uid"], r["perspective_idx"], r["doc_id"], r["modality"]
        key = key_of.get((uid, pidx, did)); dom = topic.get(key) if key else None
        if mod == "text":
            url, path = urls.get(did), None
            src = "doc_text" if dom else None
        else:
            url, path = meta.get(did, ("", None))[1], where.get(did)
            src = ("page_text" if key and key.startswith("http") else "title") if dom else None
        stats["missing"]["text_url" if mod == "text" and url is None else "image_file" if mod == "image" and path is None else "none"] += 1
        stats["by_modality"][mod] += 1; stats["by_round"][r["round"]] += 1; stats["domain_source"][src] += 1
        out.append({"uid": uid, "perspective_idx": pidx, "doc_id": did, "modality": mod, "round": r["round"], "domain": dom,
                    "domain_source": src, "local_doc_path": path, "doc_url": url})
    write_positives(_P.BENCH / "train/positives_gold_all.jsonl", out, stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("merge").set_defaults(fn=cmd_merge)
    r = sub.add_parser("round"); r.set_defaults(fn=cmd_round)
    r.add_argument("--stage1", required=True, help="glob of stage-1 outputs"); r.add_argument("--stage2", required=True)
    r.add_argument("--stage2-input", required=True, help="stage-2 judge input (sibling counts per uid)")
    r.add_argument("--modality", choices=["image", "text"], required=True); r.add_argument("--round", required=True)
    r.add_argument("--out", required=True); r.add_argument("--allow-missing", action="store_true")
    sub.add_parser("positives").set_defaults(fn=cmd_positives)
    sub.add_parser("positives-all").set_defaults(fn=cmd_positives_all)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
