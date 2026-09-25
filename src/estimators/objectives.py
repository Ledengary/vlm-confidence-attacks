from __future__ import annotations
import torch
from src.estimators.readout_bank import ATTACKED, ESTIMATORS, _f16_ste

def forward_all(adapter, ctx, pv, need_grad: bool=False):
    inp = dict(ctx.inputs)
    inp['pixel_values'] = pv.to(ctx.model_dtype)
    cm = torch.enable_grad() if need_grad else torch.no_grad()
    with cm:
        out = adapter.model(**inp, output_hidden_states=True, use_cache=False)
    return out

def constraint_from_out(ctx, out, tau_safe: float=0.0):
    L = ctx.L
    k = len(ctx.constraint_ids)
    rows = out.logits[0, L - 1:L - 1 + k].float()
    cid = torch.tensor(ctx.constraint_ids, device=rows.device)
    chosen = rows.gather(1, cid.view(-1, 1)).squeeze(1)
    masked = rows.scatter(1, cid.view(-1, 1), torch.full((k, 1), -1000000000.0, device=rows.device, dtype=rows.dtype))
    rival = masked.max(dim=1).values
    margins = chosen - rival
    min_margin = margins.min()
    return {'min_margin': min_margin, 'feasible': bool(min_margin.item() >= tau_safe)}

def objective_logit(bank, ctx, out, item_id: str, objective: str):
    L, n = (ctx.L, len(ctx.ans_ids))
    hs = out.hidden_states[-1][0]
    if objective in ('pik', 'saplma'):
        locus = L - 1 if objective == 'pik' else hs.shape[0] - 1
        h = _f16_ste(hs[locus].float())
        logit = bank.probe[objective](h.unsqueeze(0))[0]
        if bank.probe_swapped[objective]:
            return (-logit, torch.sigmoid(-logit))
        return (logit, torch.sigmoid(logit))
    if objective in ('full', 'coupled'):
        h_sap = hs[hs.shape[0] - 1].float()
        if objective == 'full':
            logit = bank.coupled_net['FULL'](h_sap.unsqueeze(0))[0]
        else:
            B = bank.coupled_projector(item_id)
            if B is None:
                raise KeyError(f'no coupled basis for {item_id}')
            logit = bank.coupled_net['COUPLED']((B.t() @ (B @ h_sap)).unsqueeze(0))[0]
        return (logit, torch.sigmoid(logit))
    if objective == 'ccpsd':
        from src.estimators.ccpsd import ccpsd_feature_tensor
        pos = list(range(L - 1, L - 1 + n))
        H = hs[pos].float()
        LG = out.logits[0][pos].float()
        tid = torch.tensor(ctx.ans_ids, device=H.device)
        feats = ccpsd_feature_tensor(H, LG, tid, bank._lm_head, bank._lm_w.float(), ctx.model_dtype)
        scaled = (feats - bank.ccpsd_mean) / bank.ccpsd_scale
        T_, nf = scaled.shape
        x = torch.zeros(1, bank.ccpsd_msl, nf, device=H.device, dtype=scaled.dtype)
        Ln = min(T_, bank.ccpsd_msl)
        x[0, :Ln] = scaled[:Ln]
        logits, _ = bank.ccpsd_clf(x)
        d = logits[0, 1] - logits[0, 0]
        if bank.ccpsd_swapped:
            return (-d, torch.sigmoid(-d))
        return (d, torch.sigmoid(d))
    raise NotImplementedError(objective)

def clean_reference_scores(bank, adapter, ctx, item_id: str, pv):
    out = forward_all(adapter, ctx, pv, need_grad=False)
    with torch.no_grad():
        sc = bank.score_all(ctx, out, item_id, want=tuple(ESTIMATORS))
        con = constraint_from_out(ctx, out, tau_safe=0.0)
    res = {k: None if v is None else float(v.detach()) for k, v in sc.items()}
    res['_min_margin'] = float(con['min_margin'])
    res['_feasible'] = bool(con['feasible'])
    return res
assert set(ATTACKED) == {'pik', 'saplma', 'ccpsd', 'full', 'coupled'}
