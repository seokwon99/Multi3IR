"""Judge-vs-human agreement of the document-support judge (production prompt, lenient) on the 300 human-labeled
items: 100 balanced (precision, 2 annotators any-support OR) + 200 raw-pool (recall, 1 annotator).

  python -m multi3ir.eval.judge_agreement --backend vllm --model Qwen/Qwen3-VL-30B-A3B-Instruct --tp 2
  python -m multi3ir.eval.judge_agreement --backend openai --model gpt-5-mini
  python -m multi3ir.eval.judge_agreement --score            # agreement / precision / recall per verdict file
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

from multi3ir import paths as _P
from multi3ir.data.stage4_judge import LENIENT, PROMPT as BASE_PROMPT

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
PROMPT = BASE_PROMPT.replace("{lenient}", LENIENT)
KEY = _P.BENCH / "eval/judge_agreement/mturk_key.jsonl"
RECALL = _P.OUTPUTS / "consistency/recall_sample_200.jsonl"
CSV_A = _P.HUMAN / "results/main_doc_support_A266AOECCJKT16.csv"
CSV_B = _P.HUMAN / "results/main_doc_support_A2FNTIPS5UNJT5.csv"
CSV_RECALL = _P.HUMAN / "results/main_doc_support_recall.csv"
IMG_DIRS = [_P.BENCH / "corpus/images", _P.STAGE3_SERPER / "serper_images_min4"]
OUT = _P.HUMAN / "soft_eval"
FIXED = {"Qwen3VL-30B-A3B (prod)": "qwen_promptsweep_verdicts_P2a_lenient_fewshot.jsonl",
         "Qwen3VL-30B Thinking": "modelsweep_thinking_lenient.jsonl",
         "GPT-5-mini": "modelsweep_gpt5mini_lenient.jsonl"}


def parse_verdict(raw):
    """Last <verdict> tag wins; else the last bare label after any </think>."""
    ms = re.findall(r"<verdict>\s*(fully|partially|not)\s*</verdict>", raw or "", re.I)
    if ms:
        return ms[-1].lower()
    labels = re.findall(r"\b(fully|partially|not)\b", (raw or "").split("</think>")[-1].lower())
    return labels[-1] if labels else "unknown"


def human_label(path):
    """item_number -> support | partial | not."""
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r["Input.ITEM_NUMBER"])] = ("support" if r.get("Answer.item_d_pi.support") == "true" else
                                            "partial" if r.get("Answer.item_d_pi.partial") == "true" else "not")
    return out


def image_path(doc_id):
    return next((str(d / doc_id) for d in IMG_DIRS if (d / doc_id).exists()), None)


def load_items():
    items, rec = [], human_label(CSV_RECALL)
    for r in map(json.loads, RECALL.open()):   # stored image_path predates the repo move: re-resolve
        ip = r.get("image_path") if r["modality"] == "text" else image_path(r["doc_id"])
        items.append({"source": "recall", "sample_id": int(r["sample_id"]), "uid": r["uid"], "doc_id": r["doc_id"],
                      "modality": r["modality"], "perspective": r["perspective"], "doc_text": r.get("doc_text"),
                      "image_path": ip, "human": rec.get(int(r["sample_id"]))})
    A, B = human_label(CSV_A), human_label(CSV_B)
    persp = {(d["uid"], d["perspective_idx"]): (d.get("summary") or "").strip() for d in map(json.loads, _P.EVAL_GOLD.open())}
    key = [json.loads(l) for l in KEY.open()]
    need = {it["doc_id"] for it in key if it["modality"] == "text"}
    doc_text = {}
    for line in (_P.BENCH / "corpus/text.jsonl").open():
        d = json.loads(line)
        if d["doc_id"] in need and d.get("text"):
            doc_text[d["doc_id"]] = d["text"]
            if len(doc_text) == len(need):
                break
    for it in key:
        sid = it["sample_id"]
        items.append({"source": "precision", "sample_id": sid, "uid": it["uid"], "doc_id": it["doc_id"],
                      "modality": it["modality"], "perspective": persp[(it["uid"], it["pi_idx"])],
                      "doc_text": doc_text.get(it["doc_id"]) if it["modality"] == "text" else None,
                      "image_path": image_path(it["doc_id"]) if it["modality"] == "image" else None,
                      "human": "support" if "support" in (A.get(sid), B.get(sid)) else "not"})
    return [it for it in items if (it["doc_text"] if it["modality"] == "text" else it["image_path"])]


def _open_image(path):
    from PIL import Image
    im = Image.open(path); im.load()
    if max(im.size) > 1024:
        im.thumbnail((1024, 1024), Image.LANCZOS)
    return im


def run_vllm(items, args):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="auto", trust_remote_code=True, max_model_len=20000 if args.think else 8192,
              tensor_parallel_size=args.tp, gpu_memory_utilization=0.9, disable_custom_all_reduce=True,
              enforce_eager=True, limit_mm_per_prompt={"image": 1})
    tok = llm.llm_engine.tokenizer
    sp = (SamplingParams(max_tokens=8192, temperature=0.6, top_p=0.95, top_k=20) if args.think
          else SamplingParams(max_tokens=256, temperature=0.0, stop=["</verdict>"]))
    reqs = []
    for it in items:
        p = PROMPT.format(perspective=it["perspective"])
        if it["modality"] == "text":
            conv = [{"role": "user", "content": f"<document>{it['doc_text'][:12000]}</document>\n\n" + p}]
            reqs.append({"prompt": tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)})
        else:
            conv = [{"role": "user", "content": [{"type": "image", "image": f"file://{os.path.abspath(it['image_path'])}"},
                                                 {"type": "text", "text": p}]}]
            reqs.append({"prompt": tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
                         "multi_modal_data": {"image": _open_image(it["image_path"])}})
    print(f"judging {len(reqs)} items with {args.model}")
    for it, o in zip(items, llm.generate(reqs, sp)):
        yield it, parse_verdict((o.outputs[0].text if o.outputs else "") + "</verdict>")


def run_openai(items, args):
    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    client = OpenAI()

    def one(it):
        p = PROMPT.format(perspective=it["perspective"])
        content = (f"<document>{it['doc_text'][:12000]}</document>\n\n" + p if it["modality"] == "text" else
                   [{"type": "text", "text": p}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                     + base64.b64encode(open(it["image_path"], "rb").read()).decode()}}])
        r = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": content}],
                                           max_completion_tokens=4000)
        return it, parse_verdict(r.choices[0].message.content)

    print(f"judging {len(items)} items with {args.model}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(one, it) for it in items]):
            yield fu.result()


def score():
    """Agr = judge-pass == human-support over all items; P on the balanced set; R on the raw-pool set."""
    files = {name: OUT / f for name, f in FIXED.items()}
    for p in sorted(glob.glob(str(OUT / "modelsweep_*.jsonl"))):
        if os.path.basename(p) not in FIXED.values():
            files[os.path.basename(p)[len("modelsweep_"):-len(".jsonl")]] = p
    print(f"{'model':<28}{'criterion':<20}{'Agr':>6}{'P':>7}{'R':>7}   (counts)\n" + "-" * 80)
    for name, path in files.items():
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        for crit, pos in (("fully only", {"fully"}), ("fully + partially", {"fully", "partially"})):
            agr = sum((r["verdict"] in pos) == (r["human"] == "support") for r in rows) / len(rows)
            pr = [r for r in rows if r["source"] == "precision" and r["verdict"] in pos]
            rc = [r for r in rows if r["source"] == "recall" and r["human"] == "support"]
            tp, rtp = sum(r["human"] == "support" for r in pr), sum(r["verdict"] in pos for r in rc)
            P, R = tp / len(pr) if pr else float("nan"), rtp / len(rc) if rc else float("nan")
            print(f"{name:<28}{crit:<20}{agr:6.2f}{P:7.2f}{R:7.2f}   P {tp}/{len(pr)}  R {rtp}/{len(rc)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "openai"])
    ap.add_argument("--model", default=None, help="hf id (vllm) or api model (openai; default gpt-5-mini-2025-08-07)")
    ap.add_argument("--tag", default=None, help="output suffix (default: from the model name)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--think", action="store_true", help="reasoning model: bigger budget + sampling (vllm)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--score", action="store_true", help="only tabulate the verdict files under human_annotate/soft_eval")
    args = ap.parse_args()
    if args.score:
        return score()
    if args.backend == "openai":
        args.model = args.model or "gpt-5-mini-2025-08-07"
    elif args.backend == "vllm" and not args.model:
        ap.error("--model is required for --backend vllm")
    elif not args.backend:
        ap.error("--backend or --score")
    tag = args.tag or args.model.split("/")[-1].replace(".", "_")
    items = load_items()
    print(f"loaded {len(items)} usable items")
    out = OUT / f"modelsweep_{tag}.jsonl"
    with out.open("w") as f:
        for it, verdict in (run_vllm if args.backend == "vllm" else run_openai)(items, args):
            f.write(json.dumps({"model": tag, "source": it["source"], "sample_id": it["sample_id"], "uid": it["uid"],
                                "doc_id": it["doc_id"], "modality": it["modality"], "verdict": verdict,
                                "human": it["human"]}) + "\n")
    print(f"wrote {out}")
    score()


if __name__ == "__main__":
    main()
