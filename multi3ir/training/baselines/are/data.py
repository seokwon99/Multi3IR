"""ARE training data: (query text, rows of its gold document embeddings) per uid, embeddings frozen and mmapped."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from multi3ir import paths as _P
from multi3ir.eval.run_eval import load_store


class DocBank:
    """Image + text embedding stores behind one row space: [0, n_img) images, then text."""

    def __init__(self, img_npz, text_npz):
        img_ids, self.img = load_store(img_npz)
        txt_ids, self.txt = load_store(text_npz)
        self.n_img, self.dim = self.img.shape[0], self.img.shape[1]
        self.id2row = {str(d): i for i, d in enumerate(img_ids)}
        self.id2row.update({str(d): self.n_img + i for i, d in enumerate(txt_ids)})

    def fetch(self, rows) -> torch.Tensor:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), self.dim), dtype=np.float32)
        img = rows < self.n_img
        if img.any():
            out[img] = self.img[rows[img]].astype(np.float32)
        if (~img).any():
            out[~img] = self.txt[rows[~img] - self.n_img].astype(np.float32)
        return torch.from_numpy(out)


class AREDataset(Dataset):
    def __init__(self, k_max: int, cap: int | None, queries=_P.TRAIN_QUERIES, positives=_P.TRAIN_POSITIVES,
                 docs_img=_P.DOC_EMB["qwen"] / "gold_docs.npz", docs_text=_P.DOC_EMB["qwen"] / "gold_docs_text.npz"):
        self.bank = DocBank(Path(docs_img), Path(docs_text))
        pairs, n_missing, n = defaultdict(list), 0, 0
        for r in map(json.loads, open(positives)):
            n += 1
            row = self.bank.id2row.get(r["doc_id"])
            if row is None:
                n_missing += 1
                continue
            pairs[r["uid"]].append((int(r["perspective_idx"]), row))
        if n_missing > 0.01 * n:
            raise RuntimeError(f"{n_missing}/{n} gold doc_ids missing from the doc bank: corpus/mapping mismatch")
        self.items = []
        for r in map(json.loads, open(queries)):
            if r["uid"] in pairs:
                self.items.append((r["query"], [row for _, row in sorted(pairs[r["uid"]])][:k_max]))
            if cap is not None and len(self.items) >= cap:
                break
        print(f"[ARE data] {len(self.items)} training queries (k_max={k_max})", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_factory(ds: AREDataset, tokenizer, k_pad: int, query_maxlen: int, format_fn):
    def _collate(batch):
        queries, rows_list = zip(*batch)
        tok = tokenizer(format_fn(list(queries)), padding=True, truncation=True, max_length=query_maxlen, return_tensors="pt")
        pos = torch.zeros((len(batch), k_pad, ds.bank.dim), dtype=torch.float32)
        mask = torch.zeros((len(batch), k_pad), dtype=torch.bool)
        for b, rows in enumerate(rows_list):
            n = min(len(rows), k_pad)
            perm = torch.randperm(len(rows))[:n].tolist()
            pos[b, :n], mask[b, :n] = ds.bank.fetch([rows[p] for p in perm]), True
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"], "pos_embs": pos, "pos_mask": mask}
    return _collate
