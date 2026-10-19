"""SPIN: m perturbations added to every token's hidden state at one frozen-encoder layer.

rank == 0: one global (m, D) table.  rank > 0: eps generated per query through a rank-r
bottleneck on the pooled h_l (eps_pool = last real token, or mean over non-sink tokens for
the bidirectional mmembed). Training forwards only layers [layer, L), once per slot.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Spin(nn.Module):
    """forward(h_l (B,T,D), attn (B,T)) -> (m, B, D) unit slot embeddings."""

    def __init__(self, encoder, m: int, layer: int, noise_std: float, rank: int = 0,
                 eps_pool: str = "last"):
        super().__init__()
        if eps_pool not in ("last", "mean_nosink"):
            raise ValueError(f"eps_pool must be 'last' or 'mean_nosink', got {eps_pool!r}")
        self.encoder, self.m, self.layer, self.rank, self.eps_pool = encoder, m, layer, rank, eps_pool
        for p in encoder.parameters():
            p.requires_grad_(False)
        D = encoder.hidden_size
        if rank == 0:
            self.eps = nn.Parameter(torch.randn(m, D) * noise_std)
        else:
            self.down = nn.Linear(D, rank)
            self.up = nn.Linear(rank, m * D)
            nn.init.normal_(self.up.weight, std=noise_std / rank ** 0.5)
            with torch.no_grad():                       # per-slot random bias: slots start apart
                self.up.bias.copy_((torch.randn(m, D) * noise_std).reshape(-1))

    def _pooled(self, h_l, attn):
        if self.eps_pool == "last":
            last = attn.shape[1] - attn.flip(dims=[1]).argmax(dim=1) - 1
            return h_l[torch.arange(h_l.shape[0], device=h_l.device), last]
        m = attn.clone()
        m[:, 0] = 0                                     # position 0 is the attention sink
        w = m.to(h_l.dtype).unsqueeze(-1)
        return (h_l * w).sum(1) / w.sum(1).clamp(min=1)

    def _eps(self, h_l, attn):
        """-> (m, B, D)."""
        if self.rank == 0:
            return self.eps.to(h_l.dtype).unsqueeze(1).expand(self.m, h_l.shape[0], -1)
        z = torch.relu(self.down(self._pooled(h_l, attn).to(self.down.weight.dtype)))
        return self.up(z).view(-1, self.m, h_l.shape[-1]).permute(1, 0, 2).to(h_l.dtype)

    def forward(self, h_l, attn):
        m, (B, T, D) = self.m, h_l.shape
        perturbed = (h_l.unsqueeze(0) + self._eps(h_l, attn)[:, :, None, :]).reshape(m * B, T, D)
        return self.encoder.encode_from_layer(perturbed, attn.repeat(m, 1), self.layer).reshape(m, B, -1)


def save(spin: Spin, path, **extra):
    """Trainable state + the shape args needed to rebuild it."""
    torch.save({"state": {k: v.detach().cpu() for k, v in spin.state_dict().items()
                          if not k.startswith("encoder.")},
                "m": spin.m, "layer": spin.layer, "rank": spin.rank, "eps_pool": spin.eps_pool,
                "eps_form": "vector", **extra}, path)


def load(path, encoder, encoder_name: str) -> Spin:
    """Rebuild a trained Spin onto `encoder`; refuses checkpoints from another backbone or form."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    trained_on = ck.get("args", {}).get("encoder")
    if trained_on != encoder_name:
        raise RuntimeError(f"{path} was trained on {trained_on!r}, loading onto {encoder_name!r}")
    if ck.get("eps_form", "vector") != "vector":
        raise RuntimeError(f"{path}: eps_form={ck['eps_form']!r} was removed 2026-09-11 "
                           f"(.archive/refactor_2026_09_11/training)")
    spin = Spin(encoder, m=ck["m"], layer=ck["layer"], noise_std=0.0, rank=ck["rank"],
                eps_pool=ck.get("eps_pool", "last"))
    missing, unexpected = spin.load_state_dict(ck["state"], strict=False)
    bad = [k for k in missing + unexpected if not k.startswith("encoder.")]
    if bad:
        raise RuntimeError(f"checkpoint does not fit this model: {bad}")
    return spin.to(encoder.device).eval()
