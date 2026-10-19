"""Stage 4a: two-stage document support judge (Qwen3-VL-30B-A3B-Instruct via vLLM), labels fully / partially / not.

    python -m multi3ir.data.stage4_judge shard --source both --n_shards 10                      # split inputs by uid, balanced per category
    python -m multi3ir.data.stage4_judge judge --stage 1 --source image --tp 4 --image_input <shard.jsonl> --output <s1.jsonl>
    python -m multi3ir.data.stage4_judge judge --stage 2 --source image --tp 4 --image_input <full search.jsonl> \
        --stage1_output <s1.jsonl> --output <s2.jsonl>

Stage 1 judges (document, own perspective); stage 2 re-judges stage-1 "fully" docs against every sibling perspective of the
same question, so --image_input/--text_input must then hold all perspectives. Data-parallel with --num_shards/--shard; resumable.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

from multi3ir import paths as _P

sys.stdout.reconfigure(line_buffering=True)
IMAGE_INPUT = _P.STAGE3_SERPER / "serper_search_min4.jsonl"
TEXT_INPUT = _P.STAGE3_C4 / "c4_dense_reranked_top10_min4.jsonl"
BM25_INPUT = _P.STAGE3_C4 / "c4_bm25_top100_min4.jsonl"
IMG_DIR = _P.STAGE3_SERPER / "serper_images_min4"
SHARDS_DIR = _P.STAGE4 / "shards"
MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MAX_NEW_TOKENS = 256

PROMPT = """You are judging whether a document supports answering a specific perspective.

Follow this reasoning process:
1. Identify what the perspective specifically asks for.
2. Describe what the document actually contains — only what is literally visible. Do NOT assume or infer content.
3. Compare the two and give a verdict.

Verdicts:
- fully: the document explicitly contains the specific answer the perspective asks for.
- partially: the document has related information but does not fully answer the specific question.
- not: the document is only topically related or unrelated, with no specific answer.

{lenient}Format your output as:
<reason>your reasoning combining steps 1-3 in one paragraph</reason>
<verdict>fully|partially|not</verdict>

Here are examples of correct reasoning:

Example 1:
<reason>The perspective asks for a method to spatially join point data to polygon attributes in QGIS. The document shows the QGIS "Join Attributes by Location" dialog with input layer, join layer configuration, and attribute selection fields. This is the exact tool and configuration needed to perform the spatial join.</reason>
<verdict>fully</verdict>

Example 2:
<reason>The perspective asks whether Chi-Running and POSE methodologies specifically teach forefoot striking technique. The document shows a generic running form infographic comparing wrong versus correct posture, illustrating mid-foot strike and cadence targets, but no methodology names are mentioned anywhere. The form concepts are related but Chi-Running and POSE are never referenced.</reason>
<verdict>partially</verdict>

Example 3:
<reason>The perspective asks for a specific PHP substr code solution for trailing comma removal. The document shows only a blog title image reading "Remove the Last Character from a String in PHP" with a decorative clipart illustration. There is no code, no substr usage, and no actual solution visible.</reason>
<verdict>not</verdict>

Example 4:
<reason>The perspective asks for a comprehensive list of IRS payment methods including online payment, direct debit, credit card, and installment agreements with usage instructions. The document shows an infographic depicting only installment agreement steps. There is no mention of online payment, direct debit, or credit card options. Only one of several requested payment methods is covered, without detailed usage instructions.</reason>
<verdict>partially</verdict>

Now judge the following:

<perspective>{perspective}</perspective>"""
LENIENT = ("Important: do not be overly strict. If the document substantively addresses what the perspective asks, "
           "choose fully. Reserve \"not\" only for documents that are clearly unrelated.\n\n")


def parse_verdict(raw):
    m = re.search(r"<verdict>\s*(fully|partially|not)\s*</verdict>", raw, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return next((l for l in ("fully", "partially", "not") if re.search(rf"\b{l}\b", raw.lower())), "unknown")


def parse_reason(raw):
    m = re.search(r"<reason>\s*(.*?)\s*</reason>", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def doc_id_of(doc):
    return doc.get("image_name", "") if doc["_modality"] == "image" else (doc.get("doc_id") or doc.get("url") or "")


def load_by_uid(by_uid, path, modality):
    """by_uid[uid][pidx] = {summary, documents}; documents tagged with _modality, concatenated across inputs."""
    for line in open(path):
        r = json.loads(line)
        e = by_uid[r["uid"]].setdefault(r["perspective_idx"], {"summary": "", "documents": []})
        e["summary"] = e["summary"] or r.get("summary") or r.get("original_summary") or ""
        for d in r.get("retrieved_images" if modality == "image" else "retrieved_documents") or []:
            d["_modality"] = modality
            e["documents"].append(d)
    return by_uid


def stage1_pairs(by_uid, top_k):
    """(uid, pidx, summary, doc, doc_id, sibling=None); top_k > 0 keeps the k best-ranked docs per perspective."""
    for uid, persps in by_uid.items():
        for pidx, e in persps.items():
            docs = sorted(e["documents"], key=lambda d: int(d.get("rank", 10**9)))[:top_k] if top_k else e["documents"]
            for doc in docs:
                if e["summary"] and doc_id_of(doc):
                    yield uid, pidx, e["summary"], doc, doc_id_of(doc), None


def stage2_pairs(stage1_paths, by_uid):
    """(uid, pidx, sibling summary, doc, doc_id, sibling pidx) for every stage-1 fully doc × sibling, in stage-1 file order."""
    fully, seen = [], set()
    for path in stage1_paths:
        for line in open(path):
            r = json.loads(line)
            key = (r["uid"], r["perspective_idx"], r["doc_id"], r.get("modality") or r["source"])
            if r["support"] == "fully" and key not in seen:
                seen.add(key); fully.append(key)
    print(f"  stage-1 fully docs: {len(fully):,}")
    lookup = {(uid, pidx, doc_id_of(d), d["_modality"]): d for uid, ps in by_uid.items() for pidx, e in ps.items()
              for d in e["documents"] if doc_id_of(d)}
    for uid, pidx, did, mod in fully:
        doc = lookup.get((uid, pidx, did, mod))
        if doc is None:
            continue
        for sib, e in by_uid[uid].items():
            if sib != pidx and e["summary"]:
                yield uid, pidx, e["summary"], doc, did, sib


def build_input(llm, doc, perspective, img_dir, max_model_len, prompt):
    """vLLM input for one (document, perspective) pair, or None if the image is missing/unreadable or the prompt is too long."""
    from PIL import Image
    if doc["_modality"] == "image":
        try:
            img = Image.open(os.path.join(img_dir, doc.get("image_name", ""))); img.load()
        except Exception:
            return None
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.LANCZOS)
        content = [{"type": "image", "image": "file://" + os.path.abspath(os.path.join(img_dir, doc["image_name"]))},
                   {"type": "text", "text": prompt.format(perspective=perspective)}]
    else:
        text = doc.get("text") or doc.get("text_preview") or ""
        if not text:
            return None
        content = f"<document>{text[:12000]}</document>\n\n" + prompt.format(perspective=perspective)
    tok = llm.llm_engine.tokenizer
    formatted = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    if len(tok.encode(formatted)) > max_model_len - MAX_NEW_TOKENS:
        return None
    return {"prompt": formatted, "multi_modal_data": {"image": img}} if doc["_modality"] == "image" else {"prompt": formatted}


def cmd_judge(a):
    prompt = PROMPT.replace("{lenient}", LENIENT if a.lenient else "")
    if a.source != "text" and not os.path.isdir(a.img_dir):
        raise FileNotFoundError(f"--img_dir {a.img_dir} does not exist")
    out = a.output or str(_P.STAGE4 / f"support_stage{a.stage}_{a.source}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    by_uid = defaultdict(dict)
    if a.source != "text":
        load_by_uid(by_uid, a.image_input, "image")
    if a.source != "image":
        load_by_uid(by_uid, a.text_input, "text")
    print(f"stage {a.stage} source {a.source}: {len(by_uid):,} questions -> {out}")
    if a.stage == 1:
        pairs = list(stage1_pairs(by_uid, a.top_k))
    else:
        s1 = a.stage1_output or str(_P.STAGE4 / f"support_stage1_{a.source}.jsonl")
        s1_paths = sorted(glob.glob(s1)) if any(c in s1 for c in "*?[") else [s1]
        if not s1_paths:
            raise FileNotFoundError(f"--stage1_output matched nothing: {s1}")
        pairs = list(stage2_pairs(s1_paths, by_uid))
    print(f"  candidate pairs: {len(pairs):,}")
    # shard before filtering done pairs: a worker's slice must be a pure function of the input
    if a.num_shards > 1:
        pairs = pairs[a.shard::a.num_shards]
        print(f"  sub-shard {a.shard}/{a.num_shards}: {len(pairs):,}")
    done = set()
    for path in [out] + [p for p in sorted(glob.glob(a.resume_glob or "")) if p != out]:
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                done.add((r["uid"], r["perspective_idx"], r["doc_id"], r.get("modality") or r["source"], r.get("sibling_perspective_idx")))
    pairs = [p for p in pairs if (p[0], p[1], p[4], p[3]["_modality"], p[5]) not in done]
    print(f"  remaining: {len(pairs):,} ({len(done):,} done)")
    if a.limit:
        pairs = pairs[:a.limit]
    if not pairs:
        print("nothing to do"); return

    from vllm import LLM, SamplingParams
    # Turing (24GB tier) has no bf16 and hangs in video profiling: --dtype float16 plus video:0 keeps it usable
    llm = LLM(model=a.model, dtype=a.dtype, limit_mm_per_prompt={"image": 0 if a.source == "text" else 1, "video": 0},
              trust_remote_code=True, max_model_len=a.max_model_len, tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.gpu_mem, max_num_batched_tokens=a.max_batched_tokens,
              disable_custom_all_reduce=True, enforce_eager=not a.cudagraph)
    sp = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0, stop=["</verdict>"])
    key = "support" if a.stage == 1 else "sibling_support"
    for b in range(0, len(pairs), a.batch_size):
        batch = pairs[b:b + a.batch_size]
        inputs, meta = [], []
        for uid, pidx, summary, doc, did, sib in batch:
            inp = build_input(llm, doc, summary, a.img_dir, a.max_model_len, prompt)
            if inp is not None:
                inputs.append(inp); meta.append((uid, pidx, summary, doc["_modality"], did, sib))
        if not inputs:
            continue
        t0 = time.time()
        outputs = llm.generate(inputs, sp)
        rows = []
        for (uid, pidx, summary, mod, did, sib), o in zip(meta, outputs):
            raw = (o.outputs[0].text.strip() if o.outputs else "") + "</verdict>"
            row = {"uid": uid, "perspective_idx": pidx, "source": a.source, "modality": mod, "doc_id": did}
            if a.stage == 1:
                row.update(support=parse_verdict(raw), reason=parse_reason(raw))
            else:
                row.update(sibling_perspective_idx=sib, sibling_summary=summary, sibling_support=parse_verdict(raw),
                           sibling_reason=parse_reason(raw))
            rows.append(row)
        with open(out, "a") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        c = Counter(r[key] for r in rows)
        print(f"  batch {b // a.batch_size + 1}/{(len(pairs) + a.batch_size - 1) // a.batch_size}: {len(rows)} pairs "
              f"{time.time()-t0:.0f}s  " + " ".join(f"{l}={c[l]}" for l in ("fully", "partially", "not", "unknown")))
    c = Counter(json.loads(l)[key] for l in open(out))
    print(f"\nsummary n={sum(c.values()):,}: " + " ".join(f"{l}={c[l]:,}" for l in ("fully", "partially", "not", "unknown")))
    print(f"Done → {out}")


def cmd_shard(a):
    """All perspectives of a uid go to one shard; uids are dealt round-robin within each category."""
    counters, uid_to_shard, breakdown = defaultdict(int), {}, defaultdict(Counter)
    for line in open(_P.PERSPECTIVES):
        d = json.loads(line)
        uid_to_shard[d["uid"]] = counters[d["category"]] % a.n_shards
        counters[d["category"]] += 1
        breakdown[uid_to_shard[d["uid"]]][d["category"]] += 1
    print(f"{len(uid_to_shard):,} questions -> {a.n_shards} shards")
    for s in range(a.n_shards):
        print(f"  shard {s:02d}: " + " ".join(f"{c}={n:,}" for c, n in sorted(breakdown[s].items())))
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, path in (("image", a.image_input), ("text", a.text_input)):
        if a.source not in (prefix, "both"):
            continue
        handles = [open(SHARDS_DIR / f"{prefix}_{i:02d}.jsonl", "w") for i in range(a.n_shards)]
        counts, missing = [0] * a.n_shards, 0
        for line in open(path):
            s = uid_to_shard.get(json.loads(line)["uid"])
            if s is None:
                missing += 1; continue
            handles[s].write(line); counts[s] += 1
        for h in handles:
            h.close()
        print(f"{prefix}: rows per shard {counts}" + (f"; {missing:,} rows with unknown uid" if missing else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard"); s.set_defaults(fn=cmd_shard)
    s.add_argument("--source", choices=["image", "text", "both"], default="both")
    s.add_argument("--n_shards", type=int, default=10)
    s.add_argument("--image_input", default=str(IMAGE_INPUT)); s.add_argument("--text_input", default=str(BM25_INPUT))
    j = sub.add_parser("judge"); j.set_defaults(fn=cmd_judge)
    j.add_argument("--stage", type=int, choices=[1, 2], required=True)
    j.add_argument("--source", choices=["image", "text", "both"], default="both")
    j.add_argument("--image_input", default=str(IMAGE_INPUT)); j.add_argument("--text_input", default=str(TEXT_INPUT))
    j.add_argument("--img_dir", default=str(IMG_DIR))
    j.add_argument("--output", default=None); j.add_argument("--stage1_output", default=None, help="stage 2: path or glob")
    j.add_argument("--resume_glob", default=None, help="also treat pairs in these files as done (after re-sharding)")
    j.add_argument("--shard", type=int, default=0); j.add_argument("--num_shards", type=int, default=1)
    j.add_argument("--top_k", type=int, default=0, help="stage 1: judge only the k best-ranked docs (0 = all)")
    j.add_argument("--limit", type=int, default=0, help="first N pairs only (testing)")
    j.add_argument("--lenient", action="store_true", help="append the leniency instruction (higher recall)")
    j.add_argument("--model", default=MODEL); j.add_argument("--tp", type=int, default=4)
    j.add_argument("--batch_size", type=int, default=3000)
    j.add_argument("--dtype", default="bfloat16", help="float16 on the 24GB Turing tier")
    j.add_argument("--gpu_mem", type=float, default=0.9); j.add_argument("--max_model_len", type=int, default=8192)
    j.add_argument("--max_batched_tokens", type=int, default=8192)
    j.add_argument("--cudagraph", action="store_true", help="enforce_eager=False (off by default: MoE crashes)")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
