"""Train ARE (PyTorch Lightning): Hungarian-matched InfoNCE over (query, gold doc embeddings).

    python -m multi3ir.training.baselines.are.train --out results/ckpts/<run> --K 5 --backbone qwen --lr 3e-5 --tau 0.01
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from multi3ir import paths as _P
from multi3ir.eval.run_eval import load_gold, merge_round_robin
from multi3ir.training.baselines.are.data import AREDataset, collate_factory
from multi3ir.training.baselines.are.model import ARE, format_queries, get_tokenizer

KS = (5, 10, 20, 50, 100)


def hungarian_infonce(q, pos, mask, tau: float):
    """q (B,K,H), pos (B,k_pad,H), mask (B,k_pad). Each query's slots are Hungarian-matched to its positives;
    every matched pair is an InfoNCE row over all positives in the batch (own unmatched ones masked out)."""
    B, K, _ = q.shape
    pos, mask = pos.to(q.device, q.dtype), mask.to(q.device)
    flat, owner, pairs, unmatched = [], [], [], []
    for b in range(B):
        own = pos[b, mask[b]]
        if own.shape[0] == 0:
            continue
        sim = (q[b] @ own.T).float()
        row, col = linear_sum_assignment((-sim).detach().cpu().numpy())
        off = sum(f.shape[0] for f in flat)
        pairs += [(b, r, off + c) for r, c in zip(row.tolist(), col.tolist())]
        unmatched += [sim[k].max().detach() for k in range(K) if k not in set(row.tolist())]
        flat.append(own)
        owner += [b] * own.shape[0]
    if not pairs:
        return q.sum() * 0.0, {"match_cos": 0.0, "unmatched_cos": 0.0}
    flat, owner = torch.cat(flat), torch.tensor(owner, device=q.device)
    bs, slots, tgt = (torch.tensor([p[i] for p in pairs], device=q.device) for i in range(3))
    qm = q[bs, slots]
    logits = (qm @ flat.T).float() / tau
    logits = logits.masked_fill((owner[None] == bs[:, None]) & (torch.arange(flat.shape[0], device=q.device)[None] != tgt[:, None]), float("-inf"))
    loss = F.cross_entropy(logits, tgt)
    with torch.no_grad():
        metrics = {"match_cos": (qm * flat[tgt]).sum(-1).mean().item(),
                   "unmatched_cos": torch.stack(unmatched).mean().item() if unmatched else 0.0}
    return loss, metrics


class ARELightning(L.LightningModule):
    def __init__(self, hp: dict):
        super().__init__()
        self.save_hyperparameters(hp)
        self.model = ARE(backbone=hp["backbone"], lora_r=hp["lora_r"], lora_alpha=hp["lora_r"],
                          in_proj_bottleneck=hp["in_proj_bottleneck"])
        self.tokenizer = get_tokenizer(self.model)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[ARE] trainable params: {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
        opt = torch.optim.AdamW(params, lr=self.hparams["lr"], weight_decay=0.01)
        total = self.hparams["num_steps"]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 - 0.9 * min(1.0, s / max(1, total)))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def _p_sample(self, step):
        hp = self.hparams
        if hp["no_scheduled_sampling"] or step < hp["ss_warmup"]:
            return 0.0
        return min(hp["ss_max_p"], hp["ss_max_p"] * (step - hp["ss_warmup"]) / max(1, hp["ss_ramp"]))

    def training_step(self, batch, _):
        K, p_s = self.hparams["K"], self._p_sample(self.global_step)
        q = self.model.forward_train(batch["input_ids"], batch["attention_mask"], batch["pos_embs"], K=K, p_sample=p_s)
        loss, m = hungarian_infonce(q, batch["pos_embs"][:, :K], batch["pos_mask"][:, :K], tau=self.hparams["tau"])
        if self.global_step % self.hparams["log_every"] == 0:
            print(f"[are K={K} step {self.global_step}/{self.hparams['num_steps']}] loss={loss.item():.4f} "
                  f"match_cos={m['match_cos']:.4f} unmatched_cos={m['unmatched_cos']:.4f} p_sample={p_s:.3f} "
                  f"lr={self.lr_schedulers().get_last_lr()[0]:.2e}", flush=True)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    # inline HC@k validation on the first val_size eval uids with gold
    def on_fit_start(self):
        hp = self.hparams
        self.val_items, self._last_val = None, -1
        if hp["val_every"] <= 0:
            return
        img, txt = (np.load(hp[k], allow_pickle=True) for k in ("val_doc_imgs", "val_doc_text"))
        self.val_doc_ids = np.concatenate([img["ids"], txt["ids"]])
        self.val_doc_emb = torch.from_numpy(np.concatenate([img["embeddings"], txt["embeddings"]])).to(self.device, torch.float16)
        queries = {}
        for d in map(json.loads, open(hp["val_queries_jsonl"])):
            queries.setdefault(d["uid"], d["query"])
        cand = list(queries)[:hp["val_size"] * 4]
        self.val_gold = load_gold(hp["val_gold"], set(cand))
        uids = [u for u in cand if u in self.val_gold][:hp["val_size"]]
        if len(uids) < hp["val_size"]:
            raise RuntimeError(f"--val-size {hp['val_size']} but only {len(uids)} uids have gold")
        self.val_items = []
        for s in range(0, len(uids), 8):     # padded per batch of 8, as at training time
            chunk = uids[s:s + 8]
            tok = self.tokenizer(format_queries(self.model, [queries[u] for u in chunk]), padding=True, truncation=True,
                                 max_length=hp["query_maxlen"], return_tensors="pt")
            self.val_items += [(u, tok["input_ids"][j], tok["attention_mask"][j]) for j, u in enumerate(chunk)]
        print(f"[val] {len(uids)} uids every {hp['val_every']} steps", flush=True)
        self._validate(0)

    def _validate(self, step):
        K, covs, t0 = self.hparams["K"], {k: [] for k in KS}, time.time()
        self.model.eval()
        with torch.no_grad():
            for uid, ids, attn in self.val_items:
                slots = F.normalize(self.model.forward_infer(ids[None].to(self.device), attn[None].to(self.device), K).float().squeeze(0), dim=-1)
                top = torch.topk(slots.to(self.val_doc_emb.dtype) @ self.val_doc_emb.t(), max(KS), dim=-1).indices.cpu().numpy()
                gold = self.val_gold[uid]
                for k in KS:
                    got = {str(self.val_doc_ids[d]) for d in merge_round_robin(top, k)}
                    covs[k].append(sum(1 for g in gold.values() if g & got) / len(gold))
        self.model.train()
        print(f"[val {step:5d}] " + "  ".join(f"HC@{k}={np.mean(v):.4f}" for k, v in covs.items())
              + f"  Δt={time.time()-t0:.1f}s", flush=True)

    def on_train_batch_end(self, *_):
        step = self.global_step
        if self.val_items is not None and step != self._last_val and (step % self.hparams["val_every"] == 0 or step == 1):
            self._last_val = step
            self._validate(step)

    def on_save_checkpoint(self, checkpoint):   # trainable weights only
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items()
                                    if "lora_" in k or ".in_proj." in k or k.endswith("in_proj.weight")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", choices=["qwen", "gme"], default="qwen")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=5000)
    ap.add_argument("--query-maxlen", type=int, default=128)
    ap.add_argument("--cap", type=int, default=None, help="cap the training set (smoke tests)")
    ap.add_argument("--no-scheduled-sampling", action="store_true")
    ap.add_argument("--ss-warmup", type=int, default=500)
    ap.add_argument("--ss-ramp", type=int, default=2000)
    ap.add_argument("--ss-max-p", type=float, default=0.8)
    ap.add_argument("--in-proj-bottleneck", type=int, default=0, help=">0: 2-layer in_proj with this width")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=0, help="0 = off")
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--val-queries-jsonl", default=str(_P.EVAL_QUERIES))
    ap.add_argument("--val-gold", default=str(_P.EVAL_GOLD))
    ap.add_argument("--val-doc-imgs", default=str(_P.DOC_EMB["qwen"] / "docs.npz"))
    ap.add_argument("--val-doc-text", default=str(_P.DOC_EMB["qwen"] / "docs_text.npz"))
    ap.add_argument("--train-doc-imgs", default=str(_P.DOC_EMB["qwen"] / "gold_docs.npz"))
    ap.add_argument("--train-doc-text", default=str(_P.DOC_EMB["qwen"] / "gold_docs_text.npz"))
    args = ap.parse_args()
    L.seed_everything(args.seed, workers=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hp = vars(args)
    print(f"[ARE] {hp}", flush=True)

    lit = ARELightning(hp)
    ds = AREDataset(k_max=max(args.K, 8), cap=args.cap, docs_img=args.train_doc_imgs, docs_text=args.train_doc_text)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True,
                        collate_fn=collate_factory(ds, lit.tokenizer, args.K, args.query_maxlen,
                                                   lambda texts: format_queries(lit.model, texts)))
    ckpt = L.callbacks.ModelCheckpoint(dirpath=str(out), filename="step{step:06d}", save_top_k=-1,
                                       every_n_train_steps=args.save_every, save_weights_only=True)
    trainer = L.Trainer(default_root_dir=str(out), max_steps=args.num_steps, accumulate_grad_batches=args.accum,
                        precision="bf16-mixed", accelerator="gpu", devices=1, logger=False, enable_progress_bar=False,
                        callbacks=[ckpt], log_every_n_steps=args.log_every)
    t0 = time.time()
    trainer.fit(lit, train_dataloaders=loader)
    print(f"[ARE] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
