"""Training cache: layer-L hidden states + perspective targets per query, and the batch sampler.

Two on-disk layouts share one interface (len, [i] -> {uid, h_l, attn, targets}, uids, layer,
all_targets): a flat repack (index.npz + h_l.u16.npy + targets.f32.npy, from precompute repack)
or a list of shard_*.pt files (mmembed / gme, from precompute hl).
"""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import torch


class FlatCache:
    def __init__(self, root: Path):
        idx = np.load(root / "index.npz", allow_pickle=True)
        self.uids = np.array([str(u) for u in idx["uids"]])
        self.oT, self.oP, self.layer = idx["offsets_T"], idx["offsets_P"], int(idx["layer"])
        self.h_l = np.load(root / "h_l.u16.npy", mmap_mode="r")        # bf16 bit patterns
        self.targets = np.load(root / "targets.f32.npy", mmap_mode="r")
        if self.h_l.shape[0] != self.oT[-1] or self.targets.shape[0] != self.oP[-1]:
            raise RuntimeError(f"{root}: index does not match array lengths")
        print(f"[data] {len(self.uids):,} queries, {self.oP[-1]:,} perspectives (flat) from {root}", flush=True)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        h = torch.from_numpy(np.asarray(self.h_l[self.oT[i]:self.oT[i + 1]])).view(torch.bfloat16)
        return {"uid": self.uids[i], "h_l": h, "attn": torch.ones(h.shape[0], dtype=torch.long),
                "targets": torch.from_numpy(np.asarray(self.targets[self.oP[i]:self.oP[i + 1]]))}

    def counts(self) -> np.ndarray:
        return np.diff(self.oP)

    def all_targets(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.targets))


class ShardCache:
    def __init__(self, root: Path):
        files = sorted(root.glob("shard_*.pt"))
        if not files:
            raise FileNotFoundError(f"no index.npz and no shard_*.pt under {root}")
        self.items = []
        for f in files:   # str(): torch<=2.2 (mmembed venv) rejects Path with mmap=True
            self.items.extend(torch.load(str(f), map_location="cpu", weights_only=False, mmap=True))
            print(f"[data] {f.name} ({len(self.items):,} queries)", flush=True)
        self.uids = np.array([str(it["uid"]) for it in self.items])
        self.layer = int((root / "layer.txt").read_text().strip())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]

    def counts(self) -> np.ndarray:
        return np.array([it["targets"].shape[0] for it in self.items])

    def all_targets(self) -> torch.Tensor:
        return torch.cat([it["targets"] for it in self.items])


def load(cache_dir) -> FlatCache | ShardCache:
    root = Path(cache_dir)
    return FlatCache(root) if (root / "index.npz").exists() else ShardCache(root)


def batches(n_items: int, batch_size: int, steps: int, seed: int = 0):
    """`steps` lists of distinct row indices (a query twice in a batch would be its own negative)."""
    if batch_size > n_items:
        raise ValueError(f"batch_size {batch_size} exceeds cache size {n_items}")
    rng = random.Random(seed)
    for _ in range(steps):
        yield rng.sample(range(n_items), batch_size)
