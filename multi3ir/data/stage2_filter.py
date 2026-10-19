"""Stage 2b: keep perspectives that are valid (MiniCheck against the answers) and unique (MiniLM cosine < 0.8).

    python -m multi3ir.data.stage2_filter validity     --input <shard.jsonl>    --output <validity.jsonl>       # GPU, one per shard
    python -m multi3ir.data.stage2_filter encode       --input <validity.jsonl> --out_dir <emb_dir>             # MiniLM embeddings of valid summaries
    python -m multi3ir.data.stage2_filter dedup        --input <validity.jsonl> --output <dedup.jsonl>          # encode + greedy dedup in one pass
    python -m multi3ir.data.stage2_filter final-filter --validity_jsonl ... --relevance_jsonl ... --emb_dir <emb_dir> --output <final.jsonl>

Adds perspective_validity (validity), rewrites perspectives + kept indices (dedup / final-filter).
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORIES = ["technology", "science", "culture_recreation", "business", "life_arts"]


def html_to_text(h):
    if not isinstance(h, str):
        return ""
    try:
        soup = BeautifulSoup(h, "html.parser")
    except Exception:
        return re.sub(r"<[^>]+>", " ", h).strip()
    for a in soup.find_all("a", href=True):
        a.replace_with(f"{a.get_text(strip=True)} ({a['href']})")
    return soup.get_text(separator=" ", strip=True)


def greedy_keep_first(emb, thr):
    """Indices kept when each row is dropped if cosine >= thr with any earlier kept row."""
    n = emb.shape[0]
    if n <= 1:
        return list(range(n))
    sims = emb @ emb.T
    kept = []
    for j in range(n):
        if all(sims[j, k] < thr for k in kept):
            kept.append(j)
    return kept


def iter_valid(path):
    """Yield (line, valid perspective indices, their summaries) per record."""
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        persp, vlab = r.get("perspectives") or [], r.get("perspective_validity") or []
        idx = [i for i, v in enumerate(vlab) if v == 1 and i < len(persp)]
        yield line, idx, [persp[i]["summary"] for i in idx]


def encode(summaries, batch_size):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MINILM).encode(summaries, batch_size=batch_size, normalize_embeddings=True,
                                              show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def mode_validity(a):
    from minicheck.minicheck import MiniCheck
    records, docs, claims, counts = [], [], [], []
    for line in open(a.input, encoding="utf-8"):
        r = json.loads(line)
        summaries = [p["summary"] for p in r.get("perspectives") or []]
        doc = " ".join(html_to_text(x) for x in r.get("Answers") or [])
        docs += [doc] * len(summaries); claims += summaries; counts.append(len(summaries)); records.append(r)
    print(f"[validity] records={len(records):,} claims={len(claims):,}", flush=True)
    scorer = MiniCheck(model_name="Bespoke-MiniCheck-7B", enable_prefix_caching=True, cache_dir="./ckpts",
                       tensor_parallel_size=1, batch_size=a.batch_size, max_model_len=a.max_model_len)
    labels = [int(x) for x in scorer.score(docs=docs, claims=claims)[0]]
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    i = 0
    with open(a.output, "w", encoding="utf-8") as out:
        for r, c in zip(records, counts):
            r["perspective_validity"] = labels[i:i + c]; i += c
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[validity] valid={sum(labels)}/{len(labels)}", flush=True)


def mode_encode(a):
    flat, offsets = [], [0]
    for _, _, sums in iter_valid(a.input):
        flat.extend(sums); offsets.append(len(flat))
    embs = encode(flat, a.encode_batch_size)
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "embeddings.npy"), embs)
    np.save(os.path.join(a.out_dir, "offsets.npy"), np.asarray(offsets, dtype=np.int64))
    print(f"[encode] {len(offsets)-1:,} records, {len(flat):,} valid summaries -> {a.out_dir}", flush=True)


def mode_dedup(a):
    lines, valid, flat = [], [], []
    for line, idx, sums in iter_valid(a.input):
        lines.append(line); valid.append(idx); flat.extend(sums)
    embs = encode(flat, a.encode_batch_size)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    n_kept = off = 0
    with open(a.output, "w", encoding="utf-8") as out:
        for line, idx in zip(tqdm(lines), valid):
            r = json.loads(line)
            kept = [idx[j] for j in greedy_keep_first(embs[off:off + len(idx)], a.sim_threshold)]
            off += len(idx); n_kept += len(kept)
            r["perspectives"] = [r["perspectives"][i] for i in kept]
            r["dedup_kept_indices"] = kept
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[dedup] kept {n_kept:,}/{len(flat):,}", flush=True)


def mode_final_filter(a):
    """valid AND relevant AND unique; embeddings from `encode` are aligned to the valid perspectives."""
    embs, offsets = np.load(os.path.join(a.emb_dir, "embeddings.npy")), np.load(os.path.join(a.emb_dir, "offsets.npy"))
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    per_cat = {}
    with open(a.validity_jsonl) as fv, open(a.relevance_jsonl) as fr, open(a.output, "w") as out:
        for ridx, (vline, rline) in enumerate(zip(fv, fr)):
            v, r = json.loads(vline), json.loads(rline)
            if v["uid"] != r["uid"]:
                raise RuntimeError(f"record order mismatch at {ridx}")
            persp, vlab, rlab = v.get("perspectives") or [], v.get("perspective_validity") or [], r.get("perspective_relevance") or []
            valid = [i for i, x in enumerate(vlab) if x == 1]
            both_mask = [i < len(rlab) and rlab[i] == 1 for i in valid]
            both = [i for i, m in zip(valid, both_mask) if m]
            emb = embs[offsets[ridx]:offsets[ridx + 1]]
            kept = [both[j] for j in greedy_keep_first(emb[np.asarray(both_mask, dtype=bool)] if valid else emb, a.sim_threshold)]
            d = per_cat.setdefault(v.get("category", "?"), dict(records=0, total=0, valid=0, both=0, kept=0))
            d["records"] += 1; d["total"] += len(persp); d["valid"] += len(valid); d["both"] += len(both); d["kept"] += len(kept)
            out.write(json.dumps({"uid": v["uid"], "category": v.get("category"), "site": v.get("site"),
                                  "Title": v.get("Title"), "Body": v.get("Body"), "Answers": v.get("Answers"),
                                  "Links": v.get("Links"), "perspectives": [persp[i] for i in kept],
                                  "kept_original_indices": kept}, ensure_ascii=False) + "\n")
    print(f"  {'category':<22} {'records':>8} {'total':>10} {'valid':>10} {'both':>10} {'kept':>10}")
    for c in CATEGORIES + sorted(set(per_cat) - set(CATEGORIES)):
        if c in per_cat:
            d = per_cat[c]
            print(f"  {c:<22} {d['records']:>8,} {d['total']:>10,} {d['valid']:>10,} {d['both']:>10,} {d['kept']:>10,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["validity", "encode", "dedup", "final-filter"])
    ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--out_dir"); ap.add_argument("--emb_dir")
    ap.add_argument("--validity_jsonl"); ap.add_argument("--relevance_jsonl")
    ap.add_argument("--sim_threshold", type=float, default=0.8)
    ap.add_argument("--encode_batch_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=32, help="validity: MiniCheck batch")
    ap.add_argument("--max_model_len", type=int, default=12288)
    a = ap.parse_args()
    need = {"validity": ["input", "output"], "encode": ["input", "out_dir"], "dedup": ["input", "output"],
            "final-filter": ["validity_jsonl", "relevance_jsonl", "emb_dir", "output"]}[a.mode]
    missing = [k for k in need if getattr(a, k) is None]
    if missing:
        raise SystemExit(f"{a.mode} needs --{' --'.join(missing)}")
    {"validity": mode_validity, "encode": mode_encode, "dedup": mode_dedup, "final-filter": mode_final_filter}[a.mode](a)


if __name__ == "__main__":
    main()
