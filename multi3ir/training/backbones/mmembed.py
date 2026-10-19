"""nvidia/MM-Embed (bidirectional Mistral + latent-attention pooler), split at one layer for SPIN."""
from __future__ import annotations
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

MODEL = "nvidia/MM-Embed"
INSTRUCTION = "Retrieve images or text relevant to the user's query."
QUERY_PREFIX = "Query: "


class MMEmbedTextSplitEncoder(nn.Module):
    """Same surface as backbones.qwen. Pooling masks the instruction tokens exactly as the
    native pooler does ("Instruct: {I}" only — not "\\nQuery: ", fixed 2026-08-25)."""

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._text = model.language_model                     # BidirectionalMistralModel
        self.layers, self.final_norm = self._text.layers, self._text.norm
        self.instr_length = len(model.preprocess_fn.tokenizer(
            f"Instruct: {INSTRUCTION}", return_tensors="pt")["input_ids"][0])

    num_layers = property(lambda self: len(self.layers))
    hidden_size = property(lambda self: self._text.config.hidden_size)
    device = property(lambda self: next(self._text.parameters()).device)

    @staticmethod
    def _state(hidden, attn):
        B, T = attn.shape
        pos = torch.arange(T, device=hidden.device)[None].expand(B, T)
        return pos, _prepare_4d_attention_mask(attn, hidden.dtype)     # pad-only, no causal

    @staticmethod
    def _call(layer, h, mask, pos):
        out = layer(h, attention_mask=mask, position_ids=pos)
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def encode_to_layer(self, inputs: dict, layer_idx: int):
        ids, attn = inputs["input_ids"].clone(), inputs["attention_mask"]
        ids[inputs["input_ids"] == self._model.config.image_token_index] = 0   # as the native forward does
        ids[inputs["input_ids"] == 32001] = 2                                   # PAD -> EOS
        h = self._text.embed_tokens(ids)
        pos, mask = self._state(h, attn)
        for layer in self.layers[:layer_idx]:
            h = self._call(layer, h, mask, pos)
        return h, attn

    def encode_from_layer(self, hidden, attn, layer_idx: int):
        pos, mask = self._state(hidden, attn)
        ckpt = os.environ.get("SPIN_GRAD_CKPT", "0") == "1" and hidden.requires_grad
        h = hidden
        for layer in self.layers[layer_idx:]:
            f = lambda x, _l=layer: self._call(_l, x, mask, pos)
            h = torch.utils.checkpoint.checkpoint(f, h, use_reentrant=False) if ckpt else f(h)
        pool_mask = attn.clone()
        pool_mask[:, :self.instr_length] = 0
        pooled = self._model.latent_attention_model(self.final_norm(h), pool_mask)
        return F.normalize(pooled.float(), dim=-1)

    def encode_full(self, inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            h, attn = self.encode_to_layer(inputs, 0)
            return self.encode_from_layer(h, attn, 0)


def make_text_tokenizer(model, max_length: int = 2048):
    """max_length must match the precompute run, or long queries truncate differently than trained."""
    processor = model.preprocess_fn
    def _tok(texts):
        prompts = [f"Instruct: {INSTRUCTION}\n{QUERY_PREFIX}{t}" for t in texts]
        feats = processor(prompts, None, return_tensors="pt", padding="longest",
                          max_length=max_length, truncation=True)
        return {k: v for k, v in feats.items() if k in ("input_ids", "attention_mask")}
    return _tok
