"""Set-coverage losses on unit vectors: slots (m, D), pos (P, D) own perspectives, neg (N, D) others'.

Both share the repulsion term  -sum_s log(1 - p_s(neg)),  p_s(x) = sigmoid((slot_s . x - bias) / tau).
noisy_or covers a positive if ANY slot fires: -log(1 - prod_s (1 - p_s(pos))).
chamfer covers by assignment, no threshold: mean_p (1 - max_s slot_s . p).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _log_miss(slots, x, tau, bias):
    return F.logsigmoid(-(slots @ x.t() - bias) / tau)      # log(1 - p), stable


def _record(parts, l_pos, l_neg, slots, pos, neg):
    if parts is not None:
        with torch.no_grad():
            for k, v in (("pos", l_pos), ("neg", l_neg),
                         ("cos_pos", (slots @ pos.t()).max(dim=0).values.mean()),
                         ("cos_neg", (slots @ neg.t()).max(dim=0).values.mean())):
                parts[k] = parts.get(k, 0.0) + float(v)


def noisy_or_loss(slots, pos, neg, tau: float, bias: float, parts: dict | None = None):
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise RuntimeError(f"empty set: pos={pos.shape[0]} neg={neg.shape[0]}")
    # log(1 - exp(x)) via expm1: exact as x -> 0- (every slot barely misses); the clamp only guards log(0)
    all_miss = _log_miss(slots, pos, tau, bias).sum(dim=0).clamp(max=-1e-30)
    l_pos = -torch.log(-torch.expm1(all_miss)).mean()
    l_neg = -_log_miss(slots, neg, tau, bias).sum(dim=0).mean()
    _record(parts, l_pos, l_neg, slots, pos, neg)
    return l_pos + l_neg


def chamfer_loss(slots, pos, neg, tau: float, bias: float, parts: dict | None = None):
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise RuntimeError(f"empty set: pos={pos.shape[0]} neg={neg.shape[0]}")
    l_pos = (1.0 - (slots @ pos.t()).max(dim=0).values).mean()
    l_neg = -_log_miss(slots, neg, tau, bias).sum(dim=0).mean()
    _record(parts, l_pos, l_neg, slots, pos, neg)
    return l_pos + l_neg


LOSSES = {"noisy_or": noisy_or_loss, "chamfer": chamfer_loss}
