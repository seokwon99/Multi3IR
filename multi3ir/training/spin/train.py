"""Train SPIN: m perturbations at one frozen-encoder layer, optimised so the union of the slots
covers every perspective of a query (noisy-OR by default). Run name = config (run_name), so a
name always says what ran.

    python -m multi3ir.training.spin.train --encoder qwen --layer 18 --m 5 --data full
"""
from __future__ import annotations
import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi3ir import paths as _P
from multi3ir.eval.run_eval import load_gold, merge_round_robin
from multi3ir.training.spin import data, encode, loss as loss_mod, model
from multi3ir.training.spin.model import Spin

RUNS, MINING = _P.CKPTS / "spin", _P.CKPTS / "spin_mining"
TRAIN_GOLD, EVAL_GOLD = _P.TRAIN_POSITIVES, _P.EVAL_GOLD
DATA_CHOICES = ["full"]                # training set (the full train split); kept as a flag for run naming
# Paper settings (Appendix C.3.2): layer 0.5L, eps ~ N(0, 0.01^2), in-batch negatives, per-backbone lr.
HEAD_DEFAULTS = {"qwen":    {"layer": 18, "data": "full", "hn": "hnoff", "noise_std": 0.01, "lr": 1e-2},
                 "gme":     {"layer": 14, "data": "full", "hn": "hnoff", "noise_std": 0.01, "lr": 1e-3},
                 "mmembed": {"layer": 16, "data": "full", "hn": "hnoff", "noise_std": 0.01, "lr": 3e-4}}
NEG_DEPTH = 2048                       # columns in a mined table (mine.py --k)
CORPUS_FILES = ("corpus_docs.npz", "corpus_docs_text.npz")   # the benchmark corpus, shared by training validation and evaluation
VAL_SET, VAL_SELECT = "val", "val/HC@10"                          # spin_best.pt selects on it
KS, TOPK = (5, 10, 20, 100), 200          # validation HC@k; per-slot candidates before the round-robin merge
# Run-name qualifiers in order; every knob that can vary is here so two configs never share a name.
QUALIFIERS = [("loss", "loss"), ("eps_rank", "r"), ("eps_pool", "pool"), ("lr", "lr"),
              ("noisy_or_tau", "tau"), ("noisy_or_bias", "bias"), ("noise_std", "std"), ("noise_rel", "nr"),
              ("batch_size", "bsz"), ("steps", "st"), ("warmup_steps", "wu"), ("grad_clip", "gc"),
              ("val_every", "ve"), ("val_size", "vs")]


def cache_path(encoder: str, layer: int) -> Path:
    """Hidden-state cache of the train queries at `layer` (precompute hl + repack)."""
    return _P.BENCH / "precompute" / _P.DOC_EMB[encoder].name / f"layer{layer}"


def ref_cache(encoder: str, data: str) -> Path:
    """The cache layer-independent work (mining, query embeddings) reads: the encoder's default layer."""
    return cache_path(encoder, HEAD_DEFAULTS[encoder]["layer"])


def cache_instruction(cache_dir, encoder: str) -> str | None:
    """The query instruction the cache's h_l were built with (written by precompute hl)."""
    if encoder == "mmembed":
        return None
    f = Path(cache_dir) / "instruction.txt"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; rebuild the cache with precompute hl")
    return f.read_text().strip()


def neg_table(encoder, data, anchor) -> Path:
    return MINING / f"{encoder}_{data}_{anchor}_k{NEG_DEPTH}.npz"


def _npy_member(path, member):
    """(offset, dtype, shape) of an uncompressed .npy inside an .npz, to stream it without unzipping."""
    import ast, struct, zipfile
    zi = zipfile.ZipFile(path).getinfo(member)
    if zi.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError(f"{path}:{member} is deflated; re-save with np.savez")
    with open(path, "rb") as f:
        f.seek(zi.header_offset)
        nlen, elen = struct.unpack("<HH", f.read(30)[26:30])
        f.seek(zi.header_offset + 30 + nlen + elen)
        magic = f.read(10)
        hlen = struct.unpack("<H", magic[8:10])[0]
        hdr = ast.literal_eval(f.read(hlen).decode("latin1"))
    return zi.header_offset + 30 + nlen + elen + 10 + hlen, np.dtype(hdr["descr"]), hdr["shape"]


def load_corpus(encoder_name: str, device, files=CORPUS_FILES):
    """Image+text document embeddings, unit-normalised, streamed straight into a GPU bf16 tensor."""
    base = _P.DOC_EMB[encoder_name]
    specs = [(base / n, *_npy_member(base / n, "embeddings.npy")) for n in files]
    N, D = sum(sp[3][0] for sp in specs), specs[0][3][1]
    emb, row = torch.empty((N, D), device=device, dtype=torch.bfloat16), 0
    for src, off, dt, (n, d) in specs:
        with open(src, "rb") as f:
            f.seek(off)
            for s in range(0, n, 65536):
                c = min(65536, n - s)
                g = torch.from_numpy(np.fromfile(f, dtype=dt, count=c * d).reshape(c, d)).to(device, torch.bfloat16)
                emb[row:row + c] = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-9).to(g.dtype)
                row += c
        print(f"[corpus] {src.name}: {n:,} rows", flush=True)
    ids = [np.fromfile(src, dtype=dt, count=shape[0], offset=o)
           for src, *_ in specs for o, dt, shape in [_npy_member(src, "ids.npy")]]
    return emb, np.concatenate(ids)


@torch.no_grad()
def cov_at_k(slots, gold_rows, doc_emb, ks=KS):
    """HC@k of a query's perspectives by the round-robin merge of its (m, D) slots' rankings."""
    top = torch.topk(slots.to(doc_emb.dtype) @ doc_emb.t(), TOPK, dim=-1).indices.cpu().numpy()
    merged = merge_round_robin(top, max(ks))
    out = {}
    for k in ks:
        got = set(int(x) for x in merged[:k])
        out[k] = float(np.mean([float(any(r in got for r in rows.tolist())) for rows in gold_rows]))
    return out


class Validator:
    """HC@k on a fixed sample of test questions; queries encoded to layer L once."""

    def __init__(self, encoder, tokenize, layer, doc_emb, doc_ids, size, device):
        self.doc_emb, self.device = doc_emb, device
        pool = encode.read_queries(_P.EVAL_QUERIES)
        gold = load_gold(EVAL_GOLD, set(pool))
        eligible = sorted(u for u in pool if u in gold)
        self.uids = [eligible[i] for i in random.Random(0).sample(range(len(eligible)), size)]
        row = {str(d): i for i, d in enumerate(doc_ids)}   # a gold doc outside the corpus stays uncovered
        self.gold_rows = {u: [torch.tensor(sorted(row[g] for g in gs if g in row)) for gs in gold[u].values()]
                          for u in self.uids}
        self.items = []
        with torch.inference_mode():
            for s in range(0, len(self.uids), 8):
                chunk = self.uids[s:s + 8]
                inputs = {k: v.to(device) for k, v in tokenize([pool[u] for u in chunk]).items()}
                h, attn = encoder.encode_to_layer(inputs, layer)
                for j, uid in enumerate(chunk):
                    keep = attn[j].bool()
                    self.items.append((uid, h[j, keep].to("cpu", torch.bfloat16).clone()))

    def _gpu(self, h):
        return h.unsqueeze(0).to(self.device, torch.bfloat16), torch.ones(1, h.shape[0], dtype=torch.long, device=self.device)

    def _cov(self, slots, uid):
        return cov_at_k(slots, self.gold_rows[uid], self.doc_emb)

    def run(self, spin) -> dict[str, float]:
        was = spin.training
        spin.eval()
        with torch.no_grad():
            cov = {u: self._cov(F.normalize(spin(*self._gpu(h)).float().squeeze(1), dim=-1), u) for u, h in self.items}
        if was:
            spin.train(); spin.encoder.eval()
        return {f"{VAL_SET}/HC@{k}": float(np.mean([cov[u][k] for u in self.uids])) for k in KS}


def resolve_defaults(args):
    """--data/--layer/--noise-std/--lr/--neg-anchor left unset come from the encoder's HEAD_DEFAULTS."""
    d = HEAD_DEFAULTS[args.encoder]
    args.data = args.data or d["data"]
    args.layer = d["layer"] if args.layer is None else args.layer
    args.noise_std = d["noise_std"] if args.noise_std is None else args.noise_std
    args.lr = d["lr"] if args.lr is None else args.lr
    if args.neg_anchor is None:
        args.neg_anchor = "none" if d["hn"] == "hnoff" else d["hn"][2:].rstrip("0123456789")
    return args


def run_name(args, parser) -> str:
    """encoder [+ non-default layer/m/data/negatives] [+ non-default qualifiers] + seed."""
    resolve_defaults(args)
    d, parts = HEAD_DEFAULTS[args.encoder], [args.encoder]
    if args.layer != d["layer"]: parts.append(f"L{args.layer}")
    if args.m != 5: parts.append(f"m{args.m}")
    if args.data != d["data"]: parts.append(args.data)
    hn = "hnoff" if args.neg_anchor == "none" else f"hn{args.neg_anchor}{args.hard_neg_k}"
    if hn != d["hn"]: parts.append(hn)
    for key, prefix in QUALIFIERS:
        v = getattr(args, key)
        if v != (d[key] if key in d else parser.get_default(key)):
            parts.append(f"{prefix}{v:g}" if isinstance(v, float) else f"{prefix}{v}")
    return "_".join(parts + [f"s{args.seed}"])


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="qwen", choices=encode.ENCODERS)
    p.add_argument("--data", default=None, choices=DATA_CHOICES, help="training set")
    p.add_argument("--layer", type=int, default=None, help="injection layer (default: HEAD_DEFAULTS)")
    p.add_argument("--m", type=int, default=5, help="number of slots")
    p.add_argument("--eps-rank", type=int, default=0, help="0 = global eps table (paper); >0 = per-query bottleneck rank")
    p.add_argument("--eps-pool", default="last", choices=["last", "mean_nosink"], help="rank>0: what the bottleneck reads")
    p.add_argument("--noise-std", type=float, default=None, help="eps init std (default: HEAD_DEFAULTS, 0.01)")
    p.add_argument("--noise-rel", type=float, default=None,
                   help="eps init scale relative to the cache's median token norm (rescales the init at startup)")
    p.add_argument("--loss", default="noisy_or", choices=sorted(loss_mod.LOSSES))
    p.add_argument("--noisy-or-tau", type=float, default=0.15)
    p.add_argument("--noisy-or-bias", type=float, default=0.5)
    p.add_argument("--neg-anchor", default=None, choices=["persp", "question", "none"],
                   help="mined-negative table (mine.py --anchor); none = in-batch negatives only")
    p.add_argument("--hard-neg-k", type=int, default=128, help="columns of the mined table to use")
    p.add_argument("--steps", type=int, default=1623)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: HEAD_DEFAULTS (3e-4 mmembed / 1e-2 qwen / 1e-3 gme)")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=200, help="0 disables validation")
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--wandb-project", default="multi3ir-spin", help="empty disables W&B")
    p.add_argument("--seed", type=int, default=0)
    return p


def _git_commit() -> str:
    g = lambda *a: subprocess.run(["git", "-C", str(_P.ROOT), *a], capture_output=True, text=True).stdout.strip()
    return g("rev-parse", "--short", "HEAD") + ("-dirty" if g("status", "--porcelain") else "")


def main():
    p = build_parser()
    args = resolve_defaults(p.parse_args())
    if args.neg_anchor == "none" and args.hard_neg_k != p.get_default("hard_neg_k"):
        raise ValueError("--hard-neg-k has no effect with --neg-anchor none")
    if args.eps_rank == 0 and args.eps_pool != p.get_default("eps_pool"):
        raise ValueError("--eps-pool has no effect at --eps-rank 0")
    if args.noise_rel is not None and args.noise_std != HEAD_DEFAULTS[args.encoder]["noise_std"]:
        raise ValueError("--noise-rel sets the init scale itself; do not pass --noise-std with it")
    cache_dir = cache_path(args.encoder, args.layer)
    if not cache_dir.is_dir():
        raise ValueError(f"no hidden-state cache at {cache_dir}: run precompute hl + repack for layer {args.layer}")
    instruction = cache_instruction(cache_dir, args.encoder)
    table_path = "" if args.neg_anchor == "none" else str(neg_table(args.encoder, args.data, args.neg_anchor))

    out = RUNS / run_name(args, p)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] {out}\n[instruction] {instruction!r}", flush=True)
    config = dict(vars(args), cache_dir=str(cache_dir), neg_table=table_path, instruction=instruction,
                  commit=_git_commit())
    args_file = out / "args.json"
    if args_file.exists():        # same name + different config = a naming bug; never clobber a run's record
        old = json.loads(args_file.read_text())
        diff = {k: (old[k], v) for k, v in vars(args).items() if k in old and old[k] != v}
        if diff:
            raise RuntimeError(f"{out} already holds a run with different args: {diff}")
        config = {**{k: v for k, v in old.items() if k not in config}, **config}
    args_file.write_text(json.dumps(config, indent=2, sort_keys=True))
    torch.manual_seed(args.seed)
    device = "cuda"
    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=out.name, config=config, dir=str(_P.LOGS))

    encoder, tokenize = encode.build(args.encoder, instruction)
    encoder.eval()
    items = data.load(cache_dir)
    if items.layer != args.layer:
        raise RuntimeError(f"--layer {args.layer} but the cache was built at layer {items.layer}")
    spin = Spin(encoder, m=args.m, layer=args.layer, noise_std=args.noise_std, rank=args.eps_rank,
                eps_pool=args.eps_pool).to(device)
    print(f"[model] {args.encoder} L{args.layer} m={args.m}, trainable "
          f"{sum(q.numel() for q in spin.parameters() if q.requires_grad)/1e6:.3f}M", flush=True)
    if args.noise_rel is not None:   # measure init ||eps|| and rescale it to rel * median token norm
        sample = [items[i] for i in np.linspace(0, len(items) - 1, 200, dtype=int)]
        med = float(np.median(np.concatenate([it["h_l"][1:].float().norm(dim=-1).numpy() for it in sample])))
        with torch.no_grad():
            cur = float(torch.cat([spin._eps(it["h_l"][None].to(device, torch.bfloat16),
                                             torch.ones(1, it["h_l"].shape[0], dtype=torch.long, device=device))
                                   .float().norm(dim=-1).flatten() for it in sample[:16]]).median())
            if not (np.isfinite(med) and med > 0 and np.isfinite(cur) and cur > 0):
                raise RuntimeError(f"degenerate norms: token median {med}, init eps {cur}")
            scale = args.noise_rel * med / cur
            for t in ([spin.eps] if spin.rank == 0 else [spin.up.weight, spin.up.bias]):
                t.mul_(scale)
        print(f"[noise-rel] token norm median {med:.1f}, init ||eps|| {cur:.1f} -> x{scale:.4f}", flush=True)
        config.update(eps_norm_target=args.noise_rel * med, eps_norm_measured=cur, measured_token_norm_median=med)
        args_file.write_text(json.dumps(config, indent=2, sort_keys=True))

    persp_bank = neg_idx = None
    if args.neg_anchor != "none":   # mined once by mine.py; training only looks the rows up
        table = np.load(table_path, allow_pickle=True)
        flat = items.all_targets()
        table_data = str(table["data"])
        for got, want, what in ((str(table["encoder"]), args.encoder, "encoder"), (table_data, args.data, "data"),
                                (str(table["anchor"]), args.neg_anchor, "anchor"), (int(table["k"]), NEG_DEPTH, "depth"),
                                (int(table["n_perspectives"]), flat.shape[0], "perspective count")):
            if got != want:
                raise RuntimeError(f"{table_path}: {what} is {got!r}, expected {want!r}")
        if not (np.array([str(u) for u in table["ids"]]) == items.uids).all():
            raise RuntimeError(f"{table_path} is not in cache order")
        neg_idx = torch.from_numpy(table["neg_idx"][:, :args.hard_neg_k].astype(np.int64)).to(device)
        persp_bank = torch.empty(flat.shape, device=device, dtype=torch.bfloat16)
        for s0 in range(0, flat.shape[0], 65536):
            persp_bank[s0:s0 + 65536] = F.normalize(flat[s0:s0 + 65536].to(device, torch.float32), dim=-1).to(torch.bfloat16)
        print(f"[hard-neg] anchor={args.neg_anchor} k={args.hard_neg_k} from {tuple(persp_bank.shape)}", flush=True)

    validator = None
    if args.val_every > 0:
        doc_emb, doc_ids = load_corpus(args.encoder, device)
        validator = Validator(encoder, tokenize, args.layer, doc_emb, doc_ids, args.val_size, device)

    best_ck, best_cov = out / "spin_best.pt", -1.0
    if best_ck.exists():           # a restart only replaces spin_best.pt with a better one
        best_cov = torch.load(best_ck, map_location="cpu", weights_only=False)["cov"][VAL_SELECT]
        print(f"[resume] existing spin_best.pt at {VAL_SELECT}={best_cov:.4f}", flush=True)
    params = [q for q in spin.parameters() if q.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr)
    warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=args.warmup_steps) \
        if args.warmup_steps > 0 else None

    for step, rows in enumerate(data.batches(len(items), args.batch_size, args.steps, args.seed), 1):
        batch = [items[r] for r in rows]
        counts = [b["targets"].shape[0] for b in batch]
        pool = F.normalize(torch.cat([b["targets"] for b in batch]).to(device).float(), dim=-1)
        offsets = np.cumsum([0] + counts)
        optim.zero_grad(set_to_none=True)
        total = 0.0
        for i, b in enumerate(batch):
            slots = F.normalize(spin(b["h_l"][None].to(device, torch.bfloat16), b["attn"][None].to(device))
                                .float().squeeze(1), dim=-1)
            keep = torch.ones(pool.shape[0], dtype=torch.bool, device=device)   # exclude own rows by uid (twins)
            for j, b2 in enumerate(batch):
                if b2["uid"] == b["uid"]:
                    keep[int(offsets[j]):int(offsets[j + 1])] = False
            neg = pool[keep]
            if neg_idx is not None:
                neg = torch.cat([neg, persp_bank[neg_idx[rows[i]]].float()])
            l = loss_mod.LOSSES[args.loss](slots, pool[int(offsets[i]):int(offsets[i + 1])], neg,
                                           tau=args.noisy_or_tau, bias=args.noisy_or_bias) / len(batch)
            l.backward()
            total += l.item()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optim.step()
        if warmup is not None and step <= args.warmup_steps:
            warmup.step()

        print(f"[step {step}] loss={total:.4f}", flush=True)
        if run:
            run.log({"train/loss": total}, step=step)

        if validator and (step % args.val_every == 0 or step == args.steps):
            cov = validator.run(spin)
            print(f"[val {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in cov.items()), flush=True)
            if run:
                run.log({f"val/{k}": v for k, v in cov.items()}, step=step)
            if cov[VAL_SELECT] > best_cov:   # HC@10 peaks before the last step; keep the best eps too
                best_cov = cov[VAL_SELECT]
                model.save(spin, best_ck, step=step, cov=cov, args=vars(args))
    model.save(spin, out / "spin.pt", step=args.steps, args=vars(args))
    if run:
        run.finish()
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
