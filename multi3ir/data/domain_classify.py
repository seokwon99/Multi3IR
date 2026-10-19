"""Stage 5b: subject-domain labels for gold documents (WebOrganizer 24 topics; nvidia 26 domains; Essential-Web FDC codes).

    python -m multi3ir.data.domain_classify topic text                                  # C4 body      -> full_topics_text.jsonl
    python -m multi3ir.data.domain_classify topic image                                 # page body    -> full_topics_image.jsonl
    python -m multi3ir.data.domain_classify topic input --input gold_all_input.jsonl --model nvidia_ml --shard 0 --num-shards 4
    python -m multi3ir.data.domain_classify title-url                                   # images whose page never loaded (url + title)
    python -m multi3ir.data.domain_classify fdc --input gold_all_input.jsonl --out fdc.jsonl   # EssentialAI/eai-distill-0.5b via vLLM

Outputs {key, topic} (or {key, fdc, ...}) under results/outputs/domain_classification/; resume reads every shard file.
"""
import argparse
import glob
import hashlib
import json
import re
import time

from multi3ir import paths as _P

D, S4, SER = _P.DOMAIN, _P.STAGE4, _P.STAGE3_SERPER


def rows(path):
    for line in open(path):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def done_keys(pattern):
    """Keys already written under any shard of this output (re-sharding must not redo labelled docs)."""
    done = set()
    for f in glob.glob(pattern):
        try:
            done |= {r["key"] for r in rows(f)}
        except FileNotFoundError:
            continue
    return done


def in_shard(key, shard, num_shards):
    return num_shards == 1 or int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % num_shards == shard


def shard_pattern(out, num_shards):
    return re.sub(r"_\d+of\d+\.jsonl$", "_*of*.jsonl", out) if num_shards > 1 else out


def load_topic_model(name):
    """(tokenizer, predict(input_ids, attention_mask) -> label ids, id2label, max_len)."""
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    if name == "weborganizer":
        model_id = "WebOrganizer/TopicClassifier-NoURL"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                                   use_memory_efficient_attention=False, unpad_inputs=False)
        return (AutoTokenizer.from_pretrained(model_id), lambda i, m: model(input_ids=i, attention_mask=m).logits.argmax(-1),
                model.config.id2label, 8192, model.eval().cuda())
    # nvidia/domain-classifier ships no custom code: a PyTorchModelHubMixin (DeBERTa body + fc head at CLS).
    # AutoModelForSequenceClassification would silently build a random head.
    import torch.nn as nn
    from huggingface_hub import PyTorchModelHubMixin

    class NvidiaDomainClassifier(nn.Module, PyTorchModelHubMixin):
        def __init__(self, config):
            super().__init__()
            self.model = AutoModel.from_pretrained(config["base_model"])
            self.dropout = nn.Dropout(config["fc_dropout"])
            self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))
            self.id2label = {int(k): v for k, v in config["id2label"].items()}

        def forward(self, input_ids, attention_mask):
            return self.fc(self.dropout(self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state))[:, 0, :]
    model_id = "nvidia/domain-classifier" if name == "nvidia" else "nvidia/multilingual-domain-classifier"
    model = NvidiaDomainClassifier.from_pretrained(model_id)
    if "fc.weight" not in dict(model.named_parameters()):
        raise RuntimeError("nvidia classifier head not loaded")
    return AutoTokenizer.from_pretrained(model_id), lambda i, m: model(i, m).argmax(-1), model.id2label, 512, model.eval().cuda()


def classify(items, out, tok, predict, id2label, max_len, batch_size):
    """Length-sorted batches under a batch*L^2 token budget (the classifier materialises full attention); per-row OOM fallback."""
    import torch
    budget = batch_size * max_len * max_len // 16
    items.sort(key=lambda kt: len(kt[1]))
    batches, buf = [], []
    for k, t in items:
        L = min(len(t) // 3 + 1, max_len)
        if buf and (len(buf) + 1) * L * L > budget:
            batches.append(buf); buf = []
        buf.append((k, t))
    batches += [buf] if buf else []

    def run(rs, ml):
        enc = tok([t for _, t in rs], truncation=True, max_length=ml, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            return predict(enc["input_ids"], enc["attention_mask"]).tolist()
    n, t0 = 0, time.time()
    with open(out, "a") as w:
        for bi, batch in enumerate(batches):
            try:
                preds, rs = run(batch, max_len), batch
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); preds, rs = [], []
                for one in batch:
                    for ml in (max_len, max_len // 2, max_len // 4, 512):
                        try:
                            preds += run([one], ml); rs.append(one); break
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
            for (k, _), p in zip(rs, preds):
                w.write(json.dumps({"key": k, "topic": id2label[int(p)]}) + "\n")
            n += len(rs)
            if bi % 50 == 0:
                w.flush(); print(f"  {n}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[done] {n} labels -> {out}")


def cmd_topic(a):
    suf = "" if a.model == "weborganizer" else f"_{a.model}"
    out = a.out or str(D / (f"full_topics_{a.part}{suf}.jsonl" if a.num_shards == 1 else f"full_topics_{a.part}{suf}_{a.shard}of{a.num_shards}.jsonl"))
    done = done_keys(shard_pattern(out, a.num_shards))
    if a.part == "text":
        items = [(r["doc_id"], r["text"]) for r in rows(D / "full_text_contents.jsonl") if r["doc_id"] not in done]
    elif a.part == "image":          # one label per unique hosting page
        page = {}
        for src in ("full_page_texts.jsonl",):
            if (D / src).exists():
                for r in rows(D / src):
                    if r.get("text"):
                        page.setdefault(r["url"], r["text"])
        urls = set(json.load(open(D / "full_image_links.json")).values())
        items = [(u, page[u]) for u in sorted(urls) if u in page and u not in done]
    else:
        if not a.input:
            raise SystemExit("topic input needs --input (jsonl of {key, text})")
        items = [(r["key"], r["text"]) for r in rows(a.input) if r["key"] not in done and r.get("text")]
    items = [(k, t[:a.max_chars] if a.max_chars else t) for k, t in items if in_shard(k, a.shard, a.num_shards)]
    print(f"{len(items)} to classify ({len(done)} done)", flush=True)
    tok, predict, id2label, cap, _ = load_topic_model(a.model)
    classify(items, out, tok, predict, id2label, min(a.max_len, cap), a.batch_size)


def cmd_title_url(a):
    """Gold images whose hosting page has no body-based label: WebOrganizer URL-aware variant on "<link>\\n\\n<title> — <source>".
    Written to a separate file (full_topics_image_titleurl.jsonl) so the two label qualities never mix."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    out = D / "full_topics_image_titleurl.jsonl"
    gold = {r["doc_id"] for r in rows(S4 / "gold_docs.jsonl") if r["modality"] == "image"}
    labelled = {r["key"] for r in rows(D / "full_topics_image.jsonl") if r.get("topic")}
    done = done_keys(str(out))
    items, seen = [], set()
    for r in rows(SER / "serper_search_min4.jsonl"):
        for im in r["retrieved_images"]:
            name, link, title = im["image_name"], im.get("link") or "", (im.get("title") or "").strip()
            if name not in gold or name in seen or name in done or link in labelled or not (link or title):
                continue
            seen.add(name)
            items.append((name, f"{link}\n\n{title}" + (f" — {im['source'].strip()}" if im.get("source", "").strip() else "")))
    print(f"[title-url] {len(items):,} to classify", flush=True)
    if not items:
        return
    model_id = "WebOrganizer/TopicClassifier"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32,
                                                               use_memory_efficient_attention=False, unpad_inputs=False).eval().cuda()
    with open(out, "a") as w:
        for i in range(0, len(items), a.batch_size):
            batch = items[i:i + a.batch_size]
            enc = tok([t for _, t in batch], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                pred = model(**enc).logits.argmax(-1)
            for (name, _), p in zip(batch, pred):
                w.write(json.dumps({"key": name, "topic": model.config.id2label[int(p)], "source": "title_url"}) + "\n")
    print(f"[done] {len(items):,} labels -> {out}")


def cmd_fdc(a):
    """EssentialAI/eai-distill-0.5b over {key, text}: {key, raw, fdc, fdc2, doctype_v2}. >30k chars -> head + middle + tail 10k."""
    import random
    from vllm import LLM, SamplingParams
    done = done_keys(shard_pattern(a.out, a.num_shards))
    items = [dict(r, text=r["text"][:a.max_chars]) for r in rows(a.input)
             if r["key"] not in done and r.get("text") and in_shard(r["key"], a.shard, a.num_shards)]
    print(f"{len(items):,} to classify ({len(done):,} done)", flush=True)
    if not items:
        return

    def chunk(t):
        if len(t) <= 30000:
            return t
        s = random.Random(0).randint(10000, max(10000, len(t) - 20000))
        return f"[beginning]\n{t[:10000]}\n[middle]\n{t[s:s+10000]}\n[end]\n{t[-10000:]}"
    llm = LLM(model="EssentialAI/eai-distill-0.5b", dtype=a.dtype, max_model_len=16384, gpu_memory_utilization=0.85, enforce_eager=True)
    tok = llm.llm_engine.tokenizer
    prompts = [tok.apply_chat_template([{"role": "system", "content": "taxonomy"}, {"role": "user", "content": chunk(r["text"])}],
                                       tokenize=False, add_generation_prompt=True) for r in items]
    outs = llm.generate(prompts, SamplingParams(max_tokens=100, temperature=0))
    with open(a.out, "a") as w:
        for r, o in zip(items, outs):
            raw = o.outputs[0].text.strip(); lines = raw.split("\n")

            def field(i, j=0):
                try:
                    v = lines[i].split(",")[j].strip()
                    return v if v and v.lower() != "skip" else None
                except IndexError:
                    return None
            w.write(json.dumps({"key": r["key"], "raw": raw, "fdc": field(0), "fdc2": field(0, 1), "doctype_v2": field(6)}) + "\n")
    print(f"[done] {len(items):,} -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("topic"); t.set_defaults(fn=cmd_topic)
    t.add_argument("part", choices=["text", "image", "input"])
    t.add_argument("--input", default=None, help="input part: jsonl of {key, text}"); t.add_argument("--out", default=None)
    t.add_argument("--model", default="weborganizer", choices=["weborganizer", "nvidia", "nvidia_ml"])
    t.add_argument("--batch-size", type=int, default=32); t.add_argument("--max-len", type=int, default=8192)
    t.add_argument("--max-chars", type=int, default=0, help="truncate documents before tokenising (label comes from the head)")
    u = sub.add_parser("title-url"); u.set_defaults(fn=cmd_title_url); u.add_argument("--batch-size", type=int, default=32)
    f = sub.add_parser("fdc"); f.set_defaults(fn=cmd_fdc)
    f.add_argument("--input", required=True); f.add_argument("--out", required=True)
    f.add_argument("--dtype", default="float16"); f.add_argument("--max-chars", type=int, default=3000)
    for p in (t, f):
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
