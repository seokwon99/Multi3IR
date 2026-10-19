"""Naive fine-tuning baseline: train the query encoder against frozen document embeddings
(InfoNCE, in-batch negatives), one perspective summary per training example.

    python -m torch.distributed.run --standalone --nproc_per_node=4 -m multi3ir.training.baselines.finetune \\
        --backend qwen --out results/ckpts/<run> --mode full --lr 1e-5 --tau 0.05 --num-steps 400 --batch-size 16
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multi3ir import paths as _P
from multi3ir.eval.run_eval import load_store
from multi3ir.training.backbones.qwen import pool_last

INSTRUCTION = "Retrieve images or text relevant to the user's query."


class LoRALinear(nn.Module):
    """y = base(x) + (x A B) * alpha/r, base frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        for q in base.parameters():
            q.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.zeros(base.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.to(x.dtype)) @ self.lora_B.to(x.dtype) * self.scale


# ── backends: build() -> (model, processor); encode() -> (B, D) with grad; lm_layers() ────────
class Qwen:
    max_length = 256
    corpus = _P.DOC_EMB["qwen"]

    def build(self):
        from multi3ir.training.backbones import qwen
        self.embedder = qwen.load_embedder(max_length=self.max_length)
        return self.embedder.model, self.embedder

    def encode(self, model, proc, texts, max_length):
        inputs = proc._preprocess_inputs([proc.format_model_input(text=t) for t in texts])
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        return pool_last(model(**inputs).last_hidden_state, inputs["attention_mask"])

    def lm_layers(self, model):
        return model.model.language_model.layers


class MMEmbed:
    max_length = 512
    corpus = _P.DOC_EMB["mmembed"]

    def build(self):
        from transformers import AutoModel
        from multi3ir.training.backbones import mmembed
        model = AutoModel.from_pretrained(mmembed.MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                          attn_implementation="sdpa")
        return model, model.preprocess_fn

    def encode(self, model, proc, texts, max_length):
        tok = proc.tokenizer
        prompts = [f"{INSTRUCTION}\nQuery: {t}{tok.eos_token or ''}" for t in texts]
        feats = proc(prompts, None, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
        feats = {k: v.to(model.device) for k, v in feats.items()}
        feats["instruction_lengths"] = torch.tensor(model.calculate_instruction_length(tok, prompts, "\nQuery: ")).to(model.device)
        return model(**feats)["hidden_states"]

    def lm_layers(self, model):
        return model.language_model.layers


class Gme:
    max_length = 512
    corpus = _P.DOC_EMB["gme"]

    def build(self):
        from multi3ir.training.backbones import gme
        gme._patch_mrope()
        from transformers import AutoModel, AutoProcessor
        return (AutoModel.from_pretrained(gme.MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True),
                AutoProcessor.from_pretrained(gme.MODEL, trust_remote_code=True))

    def encode(self, model, proc, texts, max_length):
        from multi3ir.training.backbones import gme
        inputs = proc(text=[gme.PROMPT.format(instr=INSTRUCTION, q=t) for t in texts], images=None, padding=True,
                      truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        h = model(**inputs).last_hidden_state
        return h[torch.arange(h.shape[0], device=h.device), inputs["attention_mask"].sum(dim=1) - 1]

    def lm_layers(self, model):
        return model.layers


BACKENDS = {"qwen": Qwen, "mmembed": MMEmbed, "gme": Gme}


def load_bank(store, name):
    """-> (id -> row, (N, D) fp16) over the image + text halves of a document store."""
    stem = "" if name == "bench" else f"{name}_"
    ids, emb = zip(*(load_store(store / f"{stem}{n}") for n in ("docs.npz", "docs_text.npz")))
    ids = np.concatenate([np.asarray(i).astype(str) for i in ids])
    return {x: i for i, x in enumerate(ids)}, np.concatenate([np.asarray(e, dtype=np.float16) for e in emb])


def load_train(src, queries, heldout, id2row):
    """-> list of (perspective summary, [corpus rows of its gold docs]); heldout uids excluded."""
    skip = set(u.strip() for u in open(heldout) if u.strip())
    persps = {d["uid"]: d["positives"] for d in map(json.loads, open(queries))}
    docs, n_missing, n = defaultdict(set), 0, 0
    for d in map(json.loads, open(src)):
        if d["uid"] in skip:
            continue
        n += 1
        if d["doc_id"] not in id2row:
            n_missing += 1
            continue
        docs[(d["uid"], d["perspective_idx"])].add(id2row[d["doc_id"]])
    if n_missing > 0.01 * n:
        raise RuntimeError(f"{n_missing}/{n} gold docs missing from the train bank: corpus/mapping mismatch")
    return [(persps[u][p], sorted(rows)) for (u, p), rows in docs.items()]


def load_val(val_src, val_pos, id2row, size):
    """-> (uid -> query, uid -> {perspective_idx: set(rows)}) for the first `size` uids with gold."""
    queries = {d["uid"]: d["query"] for d in map(json.loads, open(val_src))}
    gold = defaultdict(lambda: defaultdict(set))
    for r in map(json.loads, open(val_pos)):
        if r["uid"] not in queries:
            continue
        for key, items in (("image_name", r["retrieved_images"]), ("doc_id", r["retrieved_documents"])):
            for it in items:
                if it[key] not in id2row:
                    raise RuntimeError(f"val doc {it[key]!r} not in the val bank (corpus mismatch)")
                gold[r["uid"]][r["perspective_idx"]].add(id2row[it[key]])
    uids = [u for u in sorted(queries) if gold[u]][:size]
    return {u: queries[u] for u in uids}, {u: dict(gold[u]) for u in uids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--mode", choices=["full", "lora", "backbone"], default="full", help="what to unfreeze")
    ap.add_argument("--src", default=str(_P.TRAIN_POSITIVES))
    ap.add_argument("--queries", default=str(_P.TRAIN_QUERIES))
    ap.add_argument("--eval-uids", default=str(_P.HELDOUT_UIDS))
    ap.add_argument("--num-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=None, help="default: 256 qwen / 512 mmembed, gme")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--val-size", type=int, default=100, help="0 disables validation")
    ap.add_argument("--val-src", default=str(_P.EVAL_QUERIES))
    ap.add_argument("--val-pos", default=str(_P.EVAL_GOLD))
    ap.add_argument("--train-corpus", default="gold", help="<name>_docs{,_text}.npz in the backend's store; bench = docs.npz")
    ap.add_argument("--val-corpus", default="bench")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=64.0)
    args = ap.parse_args()
    backend = BACKENDS[args.backend]()
    args.max_length = args.max_length or backend.max_length

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)
    main_ = rank == 0
    log = (lambda *a: print(*a, flush=True)) if main_ else (lambda *a: None)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Two banks: training gold docs live in the gold store, validation retrieves against the bench corpus.
    id2row, bank = load_bank(backend.corpus, args.train_corpus)
    bank = torch.from_numpy(bank).pin_memory()
    train = load_train(args.src, args.queries, args.eval_uids, id2row)
    log(f"[train] {len(train):,} perspectives, {args.num_steps} steps")
    if args.val_size > 0:
        val_id2row, val_bank = load_bank(backend.corpus, args.val_corpus)
        val_queries, val_gold = load_val(args.val_src, args.val_pos, val_id2row, args.val_size)
        log(f"[val] {len(val_queries)} uids")

    model, proc = backend.build()
    model = model.cuda().train()
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    layers = backend.lm_layers(model)
    if args.mode == "backbone":
        for q in model.parameters(): q.requires_grad = False
        for q in layers.parameters(): q.requires_grad = True
    elif args.mode == "lora":
        for q in model.parameters(): q.requires_grad = False
        if hasattr(model, "latent_attention_model"):          # MM-Embed's pooler trains alongside LoRA
            for q in model.latent_attention_model.parameters(): q.requires_grad = True
        for layer in layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                lin = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, LoRALinear(lin, args.lora_r, args.lora_alpha).to(lin.weight.device, lin.weight.dtype))
    if world > 1:
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={type(layers[0])}),
                     mixed_precision=MixedPrecision(torch.bfloat16, torch.bfloat16, torch.bfloat16),
                     device_id=rank, use_orig_params=True)
    trainable = [q for q in model.parameters() if q.requires_grad]
    log(f"[optim] {sum(q.numel() for q in trainable)/1e6:.1f}M trainable params")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    def state_dict():
        if world == 1:
            return model.state_dict()
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            return model.state_dict()

    @torch.no_grad()
    def validate():
        model.eval()
        q = torch.cat([F.normalize(backend.encode(model, proc, [val_queries[u] for u in uids], args.max_length).float(), dim=-1).cpu()
                       for uids in [list(val_queries)[i:i + 8] for i in range(0, len(val_queries), 8)]])
        model.train()
        if not main_:
            return 0.0
        top = (q.cuda() @ torch.from_numpy(val_bank).cuda().float().t()).topk(10, dim=1).indices.cpu()
        covs = [np.mean([bool(rows & set(top[i].tolist())) for rows in val_gold[u].values()])
                for i, u in enumerate(val_queries)]
        torch.cuda.empty_cache()
        return float(np.mean(covs))

    rng, best, perm, bi, t0 = random.Random(args.seed + rank), 0.0, list(range(len(train))), 0, time.time()
    per_dev = max(1, args.batch_size // world)
    if args.val_size > 0:
        best = validate()
        log(f"  [val 0] HC@10={best:.4f} (baseline)")
    for step in range(1, args.num_steps + 1):
        if bi + per_dev * world > len(perm):
            rng.shuffle(perm); bi = 0
        obj = [[train[i] for i in perm[bi:bi + per_dev * world]]] if main_ else [None]
        if world > 1:
            torch.distributed.broadcast_object_list(obj, src=0, device=torch.device(f"cuda:{rank}"))
        bi += per_dev * world
        local = obj[0][rank * per_dev:(rank + 1) * per_dev]
        q = F.normalize(backend.encode(model, proc, [t for t, _ in local], args.max_length).float(), dim=-1)
        pos = torch.tensor([rng.choice(rows) for _, rows in local], device="cuda")
        if world > 1:
            qs, ps = [torch.zeros_like(q) for _ in range(world)], [torch.zeros_like(pos) for _ in range(world)]
            torch.distributed.all_gather(qs, q); torch.distributed.all_gather(ps, pos)
            qs[rank] = q
            q, pos = torch.cat(qs), torch.cat(ps)
        d = F.normalize(bank[pos.cpu()].cuda(non_blocking=True).float(), dim=-1)
        loss = F.cross_entropy(q @ d.t() / args.tau, torch.arange(q.shape[0], device="cuda"))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        log(f"[step {step:4d}] loss={loss.item():.4f}  Δt={time.time()-t0:.2f}s"); t0 = time.time()
        if args.val_size > 0 and args.val_every > 0 and (step % args.val_every == 0 or step == args.num_steps):
            cov = validate()
            improved = cov > best
            best = max(best, cov)
            log(f"  [val {step:5d}] HC@10={cov:.4f} best={best:.4f}{' *' if improved else ''}")
            state = state_dict()
            if main_:
                torch.save({"state": state, "step": step, "val_cov": cov}, Path(args.out) / "model_last.pt")
                if improved:
                    torch.save({"state": state, "step": step, "val_cov": cov}, Path(args.out) / "model_best.pt")
                    log(f"  [save] best @ step {step} HC@10={cov:.4f}")
    state = state_dict()
    if main_:
        torch.save({"state": state, "step": args.num_steps}, Path(args.out) / "model_last.pt")
        log("[done]")


if __name__ == "__main__":
    main()
