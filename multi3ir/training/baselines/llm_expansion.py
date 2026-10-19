"""LLM query-expansion baseline: SFT Qwen3-4B to list N perspectives, generate them for the eval
queries, and encode each as a query vector -> (N, m, D) npz for eval.run_eval --query-type spin.

  build-sft   queries.jsonl (train) + positives_specific (val) -> sft_train.jsonl, sft_val.jsonl
  train       TRL SFT, LoRA or full
  generate    perspectives for every eval query -> jsonl {uid, perspectives, raw}
  encode      perspectives -> (N, m, D) with Qwen3-VL-Embedding-8B
GPT-5-mini through the Batch API, same prompt: llm_expansion_gpt5.py.
"""
from __future__ import annotations
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from multi3ir import paths as _P

SYS_PROMPT = ("You are a helpful assistant that decomposes a question into diverse perspectives, "
              "each covering a different aspect.")
USER_TMPL = "List {n} perspectives covering different aspects of the following question:\nQuestion: {q}"
SFT_DIR = _P.OUTPUTS / "baselines/llm_persp"


def messages(query: str, persps: list[str] | None, n: int):
    m = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": USER_TMPL.format(n=n, q=query)}]
    if persps is not None:
        m.append({"role": "assistant", "content": "\n".join(f"{i+1}. {p}" for i, p in enumerate(persps))})
    return m


def parse_persp(text: str, n: int) -> list[str]:
    """Numbered lines -> exactly n perspectives (padded with the last one, or the raw text)."""
    out = [m.group(1).strip() for ln in text.strip().split("\n") if (m := re.match(r"^\s*\d+[.:)]\s*(.+)$", ln.strip()))]
    if not out:
        return [text.strip()] * n
    return (out + [out[-1]] * n)[:n]


def dump(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path} ({len(rows)} rows)", flush=True)


def cmd_build_sft(a):
    heldout = set(u.strip() for u in open(_P.HELDOUT_UIDS) if u.strip())
    train = [{"uid": d["uid"], "n_persp": len(d["positives"]), "messages": messages(d["query"], d["positives"], len(d["positives"]))}
             for d in map(json.loads, open(_P.TRAIN_QUERIES)) if d["uid"] not in heldout and d["positives"]]
    queries = {d["uid"]: d["query"] for d in map(json.loads, open(_P.EVAL_QUERIES))}
    persps = defaultdict(dict)
    for d in map(json.loads, open(_P.EVAL_GOLD)):
        if d["uid"] in queries and d.get("summary"):
            persps[d["uid"]].setdefault(d["perspective_idx"], d["summary"])
    val = [{"uid": u, "n_persp": len(p), "messages": messages(queries[u], [p[i] for i in sorted(p)], len(p))}
           for u, p in persps.items()]
    random.Random(a.seed).shuffle(val)
    dump(Path(a.out_dir) / "sft_train.jsonl", train)
    dump(Path(a.out_dir) / "sft_val.jsonl", val[:a.val_uids])


def cmd_train(a):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class LogLoss(TrainerCallback):   # "[step N] loss=X" lines for watch_loss.py
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                print(f"[step {state.global_step}] loss={logs['loss']:.4f}", flush=True)

    lr = a.lr or (2e-4 if a.mode == "lora" else 2e-5)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = {k: Dataset.from_list([{"messages": json.loads(l)["messages"]} for l in open(p)]) for k, p in (("train", a.train), ("val", a.val))}
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False
    peft = None
    if a.mode == "lora":
        from peft import LoraConfig
        peft = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
                    gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs, learning_rate=lr,
                    lr_scheduler_type="cosine", warmup_ratio=0.03, max_length=a.max_length, bf16=True, logging_steps=20,
                    save_strategy="steps", save_steps=500, save_total_limit=2, eval_strategy="steps", eval_steps=500,
                    report_to="none", seed=a.seed, dataloader_num_workers=2, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    optim="adamw_torch" if a.mode == "lora" else "adamw_bnb_8bit",
                    completion_only_loss=True, assistant_only_loss=False, packing=False)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["val"].select(range(min(500, len(ds["val"])))),
                         processing_class=tok, peft_config=peft, callbacks=[LogLoss()])
    trainer.train()
    trainer.save_model(a.out)
    print("[done]", a.out, flush=True)


def cmd_generate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base, use_fast=True, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    if a.mode == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"), a.ckpt)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = model.eval().cuda()
    queries = {d["uid"]: d["query"] for d in map(json.loads, open(a.src))}
    uids = list(queries)
    gen = dict(max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, do_sample=a.sample)
    if a.sample:
        gen.update(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)   # Qwen3 recommended sampling
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f, torch.no_grad():
        for s in range(0, len(uids), a.batch):
            chunk = uids[s:s + a.batch]
            prompts = [tok.apply_chat_template(messages(queries[u], None, a.n), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) for u in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            ids = model.generate(**enc, **gen)
            for j, u in enumerate(chunk):
                text = tok.decode(ids[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({"uid": u, "perspectives": parse_persp(text, a.n), "raw": text}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {s + len(chunk)}/{len(uids)}", flush=True)
    print(f"[done] -> {a.out}", flush=True)


def cmd_encode(a):
    import torch
    from multi3ir.training.backbones import qwen
    rows = [json.loads(l) for l in open(a.inp)]
    m = len(rows[0]["perspectives"])
    texts = [p for r in rows for p in r["perspectives"]]
    embed = qwen.load_embedder()
    embed.model = embed.model.cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), a.batch):
            out.append(embed.process([{"text": t} for t in texts[i:i + a.batch]], normalize=True).cpu().float().numpy())
    emb = np.concatenate(out).reshape(len(rows), m, -1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, ids=np.array([r["uid"] for r in rows]), embeddings=emb)
    print(f"[saved] {emb.shape} -> {a.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build-sft")
    s.add_argument("--out-dir", default=str(SFT_DIR))
    s.add_argument("--val-uids", type=int, default=500)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_build_sft)
    s = sub.add_parser("train")
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["lora", "full"], required=True)
    s.add_argument("--model", default="Qwen/Qwen3-4B")
    s.add_argument("--train", default=str(SFT_DIR / "sft_train.jsonl"))
    s.add_argument("--val", default=str(SFT_DIR / "sft_val.jsonl"))
    s.add_argument("--lora-r", type=int, default=16)
    s.add_argument("--lr", type=float, default=None, help="default 2e-4 lora / 2e-5 full")
    s.add_argument("--epochs", type=float, default=2.0)
    s.add_argument("--bs", type=int, default=4)
    s.add_argument("--accum", type=int, default=4)
    s.add_argument("--max-length", type=int, default=1024)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("generate")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--mode", choices=["lora", "full"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--base", default="Qwen/Qwen3-4B")
    s.add_argument("--src", default=str(_P.EVAL_QUERIES))
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--max-new", type=int, default=384)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--sample", action="store_true")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("encode")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=16)
    s.set_defaults(fn=cmd_encode)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
