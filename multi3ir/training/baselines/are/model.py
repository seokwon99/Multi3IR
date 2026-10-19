"""ARE (Auto-Regressive Embedding, Chen et al. 2025): a decoder-only LM that autoregressively emits K query embeddings (LoRA)."""
from __future__ import annotations
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from multi3ir.training.backbones.qwen import _causal_mask
from multi3ir.training.baselines.finetune import LoRALinear


class ARE(nn.Module):
    def __init__(self, backbone: str = "qwen", lora_r: int = 16, lora_alpha: float = 16.0,
                 dtype=torch.bfloat16, in_proj_bottleneck: int = 0):
        super().__init__()
        self.backbone_type, self._dtype = backbone, dtype
        if backbone == "gme":
            from multi3ir.training.backbones import gme
            raw, self.processor = gme.load_model()
            self.text_model = raw
        else:
            from multi3ir.training.backbones import qwen
            self.embedder = qwen.load_embedder()
            raw = self.embedder.model
            self.text_model = raw.model.language_model
        for p in raw.parameters():
            p.requires_grad_(False)
        tm = self.text_model
        self.embed_tokens, self.layers, self.final_norm, self.rotary_emb = tm.embed_tokens, tm.layers, tm.norm, tm.rotary_emb
        H = tm.config.hidden_size
        n = 0
        for layer in self.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name), lora_r, lora_alpha))
                n += 1
        print(f"[ARE] LoRA on {n} linears (r={lora_r}, α={lora_alpha})", flush=True)
        if in_proj_bottleneck > 0:
            self.in_proj = nn.Sequential(nn.Linear(H, in_proj_bottleneck, bias=False), nn.GELU(),
                                         nn.Linear(in_proj_bottleneck, H, bias=False))
            for m in self.in_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        else:
            self.in_proj = nn.Linear(H, H, bias=False)
            nn.init.xavier_uniform_(self.in_proj.weight)
        self.in_proj.to(dtype=dtype)

    def _forward_layers(self, hidden, attn):
        B, T, _ = hidden.shape
        pos = torch.arange(T, device=hidden.device)[None].expand(B, T)
        pos_emb = self.rotary_emb(hidden, pos[None].expand(3, B, T).contiguous())
        mask = _causal_mask(attn, hidden.dtype)
        ckpt = os.environ.get("ARE_GRAD_CKPT", "1") == "1" and hidden.requires_grad

        def call(layer, x):
            out = layer(x, position_embeddings=pos_emb, attention_mask=mask, position_ids=pos)
            return out[0] if isinstance(out, tuple) else out
        h = hidden
        for layer in self.layers:
            h = torch.utils.checkpoint.checkpoint(lambda x, _l=layer: call(_l, x), h, use_reentrant=False) if ckpt else call(layer, h)
        return self.final_norm(h)

    def forward_train(self, input_ids, attn, pos_embs, K: int, p_sample: float = 0.0):
        """Teacher forcing on the first K-1 positives (+ scheduled sampling) -> (B, K, H) unit vectors.
        Inputs are LEFT-padded so the K outputs are the last K positions."""
        B = input_ids.shape[0]
        tok = self.embed_tokens(input_ids).to(self._dtype)
        gt = pos_embs[:, :K - 1, :].to(self._dtype)
        full_attn = torch.cat([attn, torch.ones(B, K - 1, device=attn.device, dtype=attn.dtype)], 1)
        pred = F.normalize(self._forward_layers(torch.cat([tok, self.in_proj(gt)], 1), full_attn)[:, -K:].float(), dim=-1)
        if p_sample <= 0:
            return pred
        with torch.no_grad():   # per-slot Bernoulli swap of the teacher input for the model's own prediction
            use_pred = (torch.rand(B, K - 1, device=attn.device) < p_sample).unsqueeze(-1)
            chosen = torch.where(use_pred, pred[:, :K - 1].to(self._dtype), gt)
        return F.normalize(self._forward_layers(torch.cat([tok, self.in_proj(chosen)], 1), full_attn)[:, -K:].float(), dim=-1)

    @torch.no_grad()
    def forward_infer(self, input_ids, attn, K: int):
        """Full autoregressive decode -> (B, K, H)."""
        emb, out = self.embed_tokens(input_ids).to(self._dtype), []
        for k in range(K):
            vec = F.normalize(self._forward_layers(emb, attn)[:, -1].float(), dim=-1)
            out.append(vec)
            if k < K - 1:
                emb = torch.cat([emb, self.in_proj(vec.to(self._dtype)).unsqueeze(1)], 1)
                attn = torch.cat([attn, torch.ones(attn.shape[0], 1, device=attn.device, dtype=attn.dtype)], 1)
        return torch.stack(out, 1)


def get_tokenizer(model: ARE):
    tok = model.processor.tokenizer if model.backbone_type == "gme" else model.embedder.processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def format_queries(model: ARE, texts, instruction: str | None = None):
    """Chat-template the raw queries the way the backbone expects."""
    if model.backbone_type == "gme":
        from multi3ir.training.backbones import gme
        return [gme.PROMPT.format(instr=instruction or gme.INSTRUCTION, q=t) for t in texts]
    return [model.embedder.processor.apply_chat_template(model.embedder.format_model_input(text=t, instruction=instruction),
                                                         add_generation_prompt=True, tokenize=False) for t in texts]
