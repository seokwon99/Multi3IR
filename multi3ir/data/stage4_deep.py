"""Stage 4c: deep retrieval rounds for perspectives without gold — judge inputs from deeper dense ranks, stage-2 inputs, doc texts.

    python -m multi3ir.data.stage4_deep window --shard 0 --lo 10 --hi 30 --perspectives <uncovered.json> --out <judge_input.jsonl> \
        --exclude '<stage-1 glob>'                                   # dense ranks [lo, hi) recomputed from the stage-3 embedding cache
    python -m multi3ir.data.stage4_deep stage2-input --stage1 '<glob>' --search <judge inputs...> --source image --out <s2_input.jsonl>
    python -m multi3ir.data.stage4_deep texts --gold gold_text.jsonl gold_text2.jsonl --out deep_texts.jsonl   # body text of new gold docs

Then judge with stage4_judge (--lenient), build gold with `stage4_gold round`, embed with stage4_embed.
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from multi3ir import paths as _P

SHARDS, C4, DEEP = _P.STAGE4 / "shards", _P.STAGE3_C4, _P.STAGE4 / "deep_nogold"


def rows(path):
    for line in open(path):
        if line.strip():
            yield json.loads(line)


def cmd_window(a):
    """Dense order of each perspective's BM25 top-100 from the cached shard embeddings; write ranks [lo, hi) with full text.
    --check verifies the recomputed top-10 against the stage-3 dense file (query/embedding alignment guard)."""
    xx = f"{a.shard:02d}"
    chunk_dir = C4 / f"c4_rerank_chunks_min4_shard{xx}"
    want = {(p["uid"], int(p["perspective_idx"])) for p in json.load(open(a.perspectives))}
    q_rows, cands, meta = [], {}, {}
    for qi, r in enumerate(rows(SHARDS / f"text_{xx}.jsonl")):
        if (r["uid"], int(r["perspective_idx"])) in want:
            q_rows.append(qi); cands[qi] = [d["doc_id"] for d in r["retrieved_documents"]]
            meta[qi] = {k: r[k] for k in ("uid", "perspective_idx", "question", "summary")}
    print(f"[shard {xx}] {len(q_rows):,} target perspectives", flush=True)
    if not q_rows:
        open(a.out, "w").close(); return
    q_emb = np.load(chunk_dir / "queries.npz")["embeddings"]
    need = defaultdict(list)
    for qi, docs in cands.items():
        for d in docs:
            need[d].append(qi)
    scores = defaultdict(dict)
    for path in sorted(glob.glob(str(chunk_dir / "chunk_*.npz"))):
        z = np.load(path)
        hit = [(row, did) for row, did in enumerate(z["ids"].tolist()) if did in need]
        if hit:
            emb = z["embeddings"]
            for row, did in hit:
                for qi in need[did]:
                    scores[qi][did] = float(q_emb[qi] @ emb[row])
    ordered = {}
    for qi in q_rows:
        missing = [d for d in cands[qi] if d not in scores[qi]]
        if missing:
            raise RuntimeError(f"q{qi}: {len(missing)} candidates have no embedding in {chunk_dir}")
        ordered[qi] = sorted(cands[qi], key=lambda d: -scores[qi][d])
    if a.check:
        ref = {(r["uid"], int(r["perspective_idx"])): {d["doc_id"] for d in r["retrieved_documents"]}
               for r in rows(C4 / f"c4_dense_reranked_top10_min4_shard{xx}.jsonl") if (r["uid"], int(r["perspective_idx"])) in want}
        checked = agree = 0
        for qi in q_rows[:a.check]:
            key = (meta[qi]["uid"], int(meta[qi]["perspective_idx"]))
            if key not in ref:
                continue
            checked += 1
            ours, extra = set(ordered[qi][:10]), list(ref[key] - set(ordered[qi][:10]))
            # tolerate only a near-tie swap at the 10/11 boundary (re-encoded docs differ by ~1e-4)
            if ours == ref[key] or (len(extra) == 1 and extra[0] in ordered[qi][10:12]
                                    and abs(scores[qi][ordered[qi][9]] - scores[qi][extra[0]]) < 1e-3):
                agree += 1
            else:
                print(f"[check] MISMATCH {key}: ours-theirs {sorted(ours - ref[key])} theirs-ours {sorted(ref[key] - ours)}")
        print(f"[check] top-10 agreement with stage-3 dense file: {agree}/{checked}", flush=True)
        if agree < checked:
            raise SystemExit("rank alignment mismatch — refusing to write")
    judged = {(r["uid"], int(r["perspective_idx"]), r["doc_id"]) for f in glob.glob(a.exclude or "") for r in rows(f)}
    pick = {qi: [d for d in ordered[qi][a.lo:a.hi] if (meta[qi]["uid"], int(meta[qi]["perspective_idx"]), d) not in judged] for qi in q_rows}
    need_text, text = {d for ds in pick.values() for d in ds}, {}
    for path in sorted(glob.glob(str(chunk_dir / "chunk_*.jsonl"))):
        for line in open(path):
            did = line[8:line.index('"', 8)]              # {"id": "<did>", ...} — cheap pre-filter before json.loads
            if did in need_text:
                text[did] = json.loads(line)["text"]
        if len(text) == len(need_text):
            break
    if need_text - set(text):
        raise RuntimeError(f"{len(need_text - set(text))} window docs have no text in {chunk_dir}")
    with open(a.out, "w") as f:
        for qi in q_rows:
            docs = [{"rank": a.lo + r, "doc_id": d, "text": text[d]} for r, d in enumerate(pick[qi])]
            f.write(json.dumps({**meta[qi], "retrieved_documents": docs}, ensure_ascii=False) + "\n")
    print(f"[shard {xx}] {len(q_rows):,} perspectives, {sum(map(len, pick.values())):,} docs (ranks {a.lo}-{a.hi-1}) -> {a.out}")


def cmd_stage2_input(a):
    """One row per perspective of every uid with a stage-1 fully doc (summaries from queries.jsonl), carrying only that
    perspective's fully docs — stage 2 finds siblings in its input, so every perspective of the question must be present."""
    key, idk = ("retrieved_images", "image_name") if a.source == "image" else ("retrieved_documents", "doc_id")
    fully = defaultdict(set)
    for f in sorted(glob.glob(a.stage1)):
        for r in rows(f):
            if r["support"] == "fully":
                fully[(r["uid"], r["perspective_idx"])].add(r["doc_id"])
    uids = {u for u, _ in fully}
    print(f"fully pairs {sum(map(len, fully.values())):,} over {len(fully):,} perspectives / {len(uids):,} questions", flush=True)
    docs = {}
    for f in a.search:
        for r in rows(f):
            k = (r["uid"], r["perspective_idx"])
            if k in fully:
                for d in r.get(key) or []:
                    if d.get(idk) in fully[k]:
                        docs.setdefault((k[0], k[1], d[idk]), d)
    missing = sum(1 for k, v in fully.items() for d in v if (k[0], k[1], d) not in docs)
    if missing:
        raise SystemExit(f"{missing} fully docs not found in --search files — wrong inputs?")
    n = 0
    with open(a.out, "w") as w:
        for q in rows(_P.TRAIN_QUERIES):
            if q["uid"] in uids:
                for i, p in enumerate(q.get("positives") or []):
                    w.write(json.dumps({"uid": q["uid"], "perspective_idx": i, "question": q.get("query", ""), "summary": p,
                                        key: [docs[(q["uid"], i, d)] for d in sorted(fully.get((q["uid"], i), ()))]},
                                       ensure_ascii=False) + "\n"); n += 1
    print(f"[done] {n:,} rows -> {a.out}")


def cmd_texts(a):
    """Body text for the doc_ids in --gold, from the stage-3 chunk files (flat {id, text}) or judge inputs (nested)."""
    need = {r["doc_id"] for g in a.gold for r in rows(g if "/" in g else DEEP / g)}
    for g in a.skip_have:
        need -= {r["doc_id"] for r in rows(g if "/" in g else DEEP / g)}
    files = sorted(p for g in a.pools for p in glob.glob(g))
    print(f"need {len(need):,} texts from {len(files)} pool files", flush=True)
    out = a.out if "/" in a.out else str(DEEP / a.out)
    found = 0
    with open(out, "w") as w:
        for f in files:
            for d in rows(f):
                for r in d.get("retrieved_documents", [{"doc_id": d.get("id", d.get("doc_id")), "text": d.get("text")}]):
                    if r.get("doc_id") in need and r.get("text"):
                        w.write(json.dumps({"doc_id": r["doc_id"], "text": r["text"]}) + "\n")
                        need.discard(r["doc_id"]); found += 1
            if not need:
                break
    print(f"[done] found {found:,}, missing {len(need):,} -> {out}")
    if need:
        json.dump(sorted(need), open(out.replace(".jsonl", "") + "_missing.json", "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window"); w.set_defaults(fn=cmd_window)
    w.add_argument("--shard", type=int, required=True); w.add_argument("--lo", type=int, default=20); w.add_argument("--hi", type=int, default=30)
    w.add_argument("--perspectives", required=True, help="json list of {uid, perspective_idx}")
    w.add_argument("--out", required=True); w.add_argument("--check", type=int, default=200, help="perspectives to verify (0 = skip)")
    w.add_argument("--exclude", default=None, help="glob of stage-1 outputs whose pairs are skipped")
    s = sub.add_parser("stage2-input"); s.set_defaults(fn=cmd_stage2_input)
    s.add_argument("--stage1", required=True); s.add_argument("--search", nargs="+", required=True)
    s.add_argument("--source", choices=["image", "text"], required=True); s.add_argument("--out", required=True)
    t = sub.add_parser("texts"); t.set_defaults(fn=cmd_texts)
    t.add_argument("--gold", nargs="+", default=["gold_text.jsonl", "gold_text2.jsonl"], help="names under deep_nogold/ or paths")
    t.add_argument("--pools", nargs="+", default=[str(C4 / "c4_rerank_chunks_min4_shard*/chunk_*.jsonl")])
    t.add_argument("--out", default="deep_texts.jsonl"); t.add_argument("--skip-have", nargs="*", default=[])
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
