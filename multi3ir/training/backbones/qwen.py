"""Qwen3-VL-Embedding-8B, text-only, split at one decoder layer for SPIN."""
from __future__ import annotations
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from multi3ir import paths as _P

MODEL = "Qwen/Qwen3-VL-Embedding-8B"
INSTRUCTION = "Retrieve images or text relevant to the user's query."   # model-card query instruction


def load_embedder(model: str = MODEL, max_length: int = 256, attn: str = "sdpa"):
    """The upstream Qwen3VLEmbedder (its package is a checkout, not a pip install)."""
    if str(_P.QWEN_SRC) not in sys.path:
        sys.path.append(str(_P.QWEN_SRC))          # the one place the upstream repo is put on the path
    from models.qwen3_vl_embedding import Qwen3VLEmbedder
    return Qwen3VLEmbedder(model, max_length=max_length, dtype=torch.bfloat16, attn_implementation=attn)


def _causal_mask(attn: torch.Tensor, dtype) -> torch.Tensor:
    B, T = attn.shape
    m = torch.triu(torch.full((T, T), torch.finfo(dtype).min, device=attn.device, dtype=dtype), 1)
    m = m[None, None].expand(B, 1, T, T).clone()
    return m.masked_fill((attn == 0)[:, None, None, :], torch.finfo(dtype).min)


def pool_last(h: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Hidden state at the last real token, whichever side is padded."""
    col = attn.shape[1] - attn.flip(dims=[1]).argmax(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), col]


class Qwen3VLTextSplitEncoder(nn.Module):
    """encode_to_layer(inputs, L) -> (h_L, attn); encode_from_layer(h_L, attn, L) -> unit embeddings."""

    def __init__(self, embedder):
        super().__init__()
        self.backbone = embedder.model                       # Qwen3VLForEmbedding
        self._text = self.backbone.model.language_model
        self.layers, self.final_norm = self._text.layers, self._text.norm

    num_layers = property(lambda self: len(self.layers))
    hidden_size = property(lambda self: self._text.config.hidden_size)
    device = property(lambda self: self.backbone.device)

    def _state(self, hidden, attn):
        B, T = attn.shape
        pos = torch.arange(T, device=hidden.device)[None].expand(B, T)
        pos_emb = self._text.rotary_emb(hidden, pos[None].expand(3, B, T).contiguous())
        return pos, pos_emb, _causal_mask(attn, hidden.dtype)

    @torch.no_grad()
    def encode_to_layer(self, inputs: dict, layer_idx: int):
        attn = inputs["attention_mask"]
        h = self._text.embed_tokens(inputs["input_ids"])
        pos, pos_emb, mask = self._state(h, attn)
        for layer in self.layers[:layer_idx]:
            h = layer(h, position_embeddings=pos_emb, attention_mask=mask, position_ids=pos)
        return h, attn

    def encode_from_layer(self, hidden, attn, layer_idx: int):
        pos, pos_emb, mask = self._state(hidden, attn)
        ckpt = os.environ.get("SPIN_GRAD_CKPT", "0") == "1" and hidden.requires_grad
        h = hidden
        for layer in self.layers[layer_idx:]:
            f = lambda x, _l=layer: _l(x, position_embeddings=pos_emb, attention_mask=mask, position_ids=pos)
            h = torch.utils.checkpoint.checkpoint(f, h, use_reentrant=False) if ckpt else f(h)
        return F.normalize(pool_last(self.final_norm(h), attn).float(), dim=-1)

    def encode_full(self, inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            h, attn = self.encode_to_layer(inputs, 0)
            return self.encode_from_layer(h, attn, 0)


def make_text_tokenizer(embedder, instruction: str | None = None):
    """list[str] -> {input_ids, attention_mask} through the embedder's own chat template."""
    def _tok(texts):
        convs = [embedder.format_model_input(text=t, instruction=instruction) for t in texts]
        return {k: v for k, v in embedder._preprocess_inputs(convs).items()
                if k in ("input_ids", "attention_mask")}
    return _tok
