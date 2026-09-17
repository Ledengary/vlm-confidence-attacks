from __future__ import annotations
import contextlib
import torch
from src.inference.engine import fresh_decode as _orig_fresh_decode
from src.estimators.objectives import constraint_from_out as _orig_constraint
from src.estimators.objectives import forward_all
from src.attacks import gate_v12
KL_SENTINEL = '\x00__KL_VIOLATION__\x00'

def stash_clean(ctx, out_clean) -> None:
    L, n = (ctx.L, len(ctx.ans_ids))
    rows = out_clean.logits[0, L - 1:L - 1 + n].float()
    ctx._kl_clean_logp = torch.log_softmax(rows, dim=-1).detach()
    ctx._kl_n = n

def kl_per_position(ctx, out) -> torch.Tensor:
    L, n = (ctx.L, ctx._kl_n)
    rows = out.logits[0, L - 1:L - 1 + n].float()
    logp_adv = torch.log_softmax(rows, dim=-1)
    logp_clean = ctx._kl_clean_logp
    p_clean = logp_clean.exp()
    return (p_clean * (logp_clean - logp_adv)).sum(dim=-1)

def kl_max(ctx, out) -> torch.Tensor:
    return kl_per_position(ctx, out).max()

@contextlib.contextmanager
def kl_enforced(adapter, delta: float):

    def wrapped_constraint(ctx, out, tau_safe: float=0.0):
        r = _orig_constraint(ctx, out, tau_safe)
        if delta == float('inf') or not hasattr(ctx, '_kl_clean_logp'):
            return r
        kl = kl_max(ctx, out)
        r = dict(r)
        r['kl'] = float(kl.detach())
        r['feasible'] = bool(r['feasible'] and kl.item() <= delta)
        return r

    def wrapped_fresh_decode(ad, ctx, pv):
        text, ids = _orig_fresh_decode(ad, ctx, pv)
        if delta == float('inf') or not hasattr(ctx, '_kl_clean_logp'):
            if text == ctx.row.get('committed_clean_answer') and getattr(ctx, '_kl_accepted', None) is None:
                with torch.no_grad():
                    o = forward_all(ad, ctx, pv, need_grad=False)
                    kp = kl_per_position(ctx, o)
                ctx._kl_accepted = {'kl_max': round(float(kp.max()), 6), 'kl_mean': round(float(kp.mean()), 6)}
            return (text, ids)
        with torch.no_grad():
            o = forward_all(ad, ctx, pv, need_grad=False)
            kp = kl_per_position(ctx, o)
            kl = float(kp.max().item())
        if kl > delta:
            return (KL_SENTINEL + (text or ''), ids)
        if text == ctx.row.get('committed_clean_answer') and getattr(ctx, '_kl_accepted', None) is None:
            ctx._kl_accepted = {'kl_max': round(kl, 6), 'kl_mean': round(float(kp.mean()), 6)}
            if getattr(ctx, '_kl_capture', False):
                L, n = (ctx.L, ctx._kl_n)
                ctx._kl_accepted_logits = o.logits[0, L - 1:L - 1 + n].float().half().cpu()
        return (text, ids)
    old_c, old_d = (gate_v12.constraint_from_out, gate_v12.fresh_decode)
    gate_v12.constraint_from_out = wrapped_constraint
    gate_v12.fresh_decode = wrapped_fresh_decode
    try:
        yield
    finally:
        gate_v12.constraint_from_out = old_c
        gate_v12.fresh_decode = old_d

def realized_kl(adapter, ctx, pv) -> dict:
    with torch.no_grad():
        o = forward_all(adapter, ctx, pv, need_grad=False)
        kp = kl_per_position(ctx, o)
    return {'kl_max': round(float(kp.max().item()), 6), 'kl_mean': round(float(kp.mean().item()), 6)}
