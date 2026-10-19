"""GME-Qwen2-VL-7B (28 layers, D=3584), text-only, split at one layer for SPIN."""
from __future__ import annotations
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from multi3ir import paths as _P
from multi3ir.training.backbones.qwen import _causal_mask, pool_last

MODEL = str(_P.CKPTS / "gme_qwen2vl_7b_clean")   # local copy: the HF repo's remote code breaks on transformers>=4.48
INSTRUCTION = "Retrieve images or text relevant to the user's query."   # model-card query instruction
PROMPT = ("<|im_start|>system\n{instr}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
          "<|im_start|>assistant\n<|endoftext|>")


def _patch_mrope():
    """transformers>=4.48 dropped the 'mrope' rope type GME's config names."""
    import transformers.modeling_rope_utils as ru
    ru.ROPE_INIT_FUNCTIONS.setdefault("mrope", ru.ROPE_INIT_FUNCTIONS["default"])
    if hasattr(ru, "ROPE_VALIDATION_FUNCTIONS"):
        ru.ROPE_VALIDATION_FUNCTIONS.setdefault(
            "mrope", ru.ROPE_VALIDATION_FUNCTIONS.get("default", lambda *a, **k: None))


def positions_from_mask(attn: torch.Tensor) -> torch.Tensor:
    """GME pads LEFT: pin the first real token at RoPE position 0 on both sides of the split."""
    return (attn.long().cumsum(dim=-1) - 1).clamp(min=0)


def load_model(model: str = MODEL):
    """-> (Qwen2VLModel, processor), frozen bf16 on cuda."""
    _patch_mrope()
    from transformers import AutoModel, AutoProcessor
    m = AutoModel.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    return m, AutoProcessor.from_pretrained(model, trust_remote_code=True)


class GmeTextSplitEncoder(nn.Module):
    """Same surface as backbones.qwen."""

    def __init__(self, model):
        super().__init__()
        inner = model if hasattr(model, "layers") else model.model
        self.layers, self.final_norm = inner.layers, inner.norm
        self.embed_tokens, self.rotary_emb, self._config = inner.embed_tokens, inner.rotary_emb, inner.config

    num_layers = property(lambda self: len(self.layers))
    hidden_size = property(lambda self: self._config.hidden_size)
    device = property(lambda self: next(self.parameters()).device)

    def _state(self, hidden, attn):
        B, T = attn.shape
        pos = positions_from_mask(attn)
        pos_emb = self.rotary_emb(hidden, pos[None].expand(3, B, T).contiguous())
        return pos, pos_emb, _causal_mask(attn, hidden.dtype)

    @staticmethod
    def _call(layer, h, pos_emb, mask, pos):
        out = layer(h, position_embeddings=pos_emb, attention_mask=mask, position_ids=pos)
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def encode_to_layer(self, inputs: dict, layer_idx: int):
        attn = inputs["attention_mask"]
        h = self.embed_tokens(inputs["input_ids"])
        pos, pos_emb, mask = self._state(h, attn)
        for layer in self.layers[:layer_idx]:
            h = self._call(layer, h, pos_emb, mask, pos)
        return h, attn

    def encode_from_layer(self, hidden, attn, layer_idx: int):
        pos, pos_emb, mask = self._state(hidden, attn)
        ckpt = os.environ.get("SPIN_GRAD_CKPT", "0") == "1" and hidden.requires_grad
        h = hidden
        for layer in self.layers[layer_idx:]:
            f = lambda x, _l=layer: self._call(_l, x, pos_emb, mask, pos)
            h = torch.utils.checkpoint.checkpoint(f, h, use_reentrant=False) if ckpt else f(h)
        return F.normalize(pool_last(self.final_norm(h), attn).float(), dim=-1)

    def encode_full(self, inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            h, attn = self.encode_to_layer(inputs, 0)
            return self.encode_from_layer(h, attn, 0)


def make_text_tokenizer(processor, max_length: int = 2048, instruction: str = INSTRUCTION):
    """`instruction` is the system prompt in front of the query: pass the cache's instruction.txt."""
    def _tok(texts):
        msgs = [PROMPT.format(instr=instruction, q=t) for t in texts]
        return processor(text=msgs, images=None, padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt")
    return _tok
