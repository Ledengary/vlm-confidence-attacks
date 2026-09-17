from __future__ import annotations
import hashlib
from dataclasses import asdict, dataclass
import torch
from src.attacks.attack import project, quantized_uint8
from src.inference.engine import fresh_decode
from src.estimators.readout_bank import ESTIMATORS
from src.estimators.objectives import constraint_from_out, forward_all, objective_logit
EPS_DEFAULT = 8.0 / 255.0

@dataclass
class GateConfig:
    label: str
    optimizer: str = 'sign'
    steps: int = 30
    n_restarts: int = 2
    eps: float = EPS_DEFAULT
    tau_safe: float = 0.0
    alpha0: float = 2.0 / 255.0
    alpha_min: float = 0.25 / 255.0
    checkpoints: tuple = (15, 30, 45)
    improve_tol: float = 0.0001
    keep_per_restart: int = 5
    random_start_max_shrink: int = 4
    adam_lr: float = 1.0 / 255.0
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-08
    adam_weight_decay: float = 0.0
    traj_every: int = 1

    def to_json(self):
        d = asdict(self)
        d['checkpoints'] = list(d['checkpoints'])
        d['adam_betas'] = list(d['adam_betas'])
        return d
CONFIGS = {'A_15x1': GateConfig(label='A_15x1', optimizer='sign', steps=15, n_restarts=1), 'B_30x2': GateConfig(label='B_30x2', optimizer='sign', steps=30, n_restarts=2), 'C_60x3': GateConfig(label='C_60x3', optimizer='sign', steps=60, n_restarts=3), 'ADAM_30x2': GateConfig(label='ADAM_30x2', optimizer='adam', steps=30, n_restarts=2)}

def restart_seed(item_id: str, objective: str, direction: str, restart: int) -> int:
    h = hashlib.md5(f'{item_id}|{objective}|{direction}|{restart}'.encode()).hexdigest()
    return 23 ^ int(h, 16) & 2147483647

class CandidatePool:

    def __init__(self, k: int):
        self.k = k
        self.items = []

    def offer(self, harm, step, score, min_margin, raw01):
        if len(self.items) >= self.k and harm <= self.items[-1][0]:
            return
        self.items.append((harm, step, score, min_margin, quantized_uint8(raw01).cpu()))
        self.items.sort(key=lambda t: -t[0])
        del self.items[self.k:]

def _eval_point(adapter, bank, ctx, pp, x, objective, item_id, tau_safe, need_grad):
    out = forward_all(adapter, ctx, pp(x), need_grad=need_grad)
    loss, score = objective_logit(bank, ctx, out, item_id, objective)
    con = constraint_from_out(ctx, out, tau_safe=tau_safe)
    return (loss, score, con)

def _random_feasible_start(adapter, bank, ctx, pp, x0, cfg, seed, objective, item_id):
    g = torch.Generator(device=x0.device)
    g.manual_seed(seed)
    d = (torch.rand(x0.shape, generator=g, device=x0.device, dtype=x0.dtype) * 2.0 - 1.0) * cfg.eps
    for shrink in range(cfg.random_start_max_shrink + 1):
        x = project(x0 + d / 2 ** shrink, x0, cfg.eps)
        _, _, con = _eval_point(adapter, bank, ctx, pp, x, objective, item_id, cfg.tau_safe, False)
        if con['feasible']:
            return (x.detach(), shrink, True)
    return (x0.clone(), cfg.random_start_max_shrink + 1, False)

def run_direction(adapter, bank, pp, ctx, item_id: str, objective: str, direction: str, cfg: GateConfig, official_pv_fn=None, ecdf=None) -> dict:
    dev = ctx.raw01.device
    x0 = ctx.raw01.detach().to(torch.float32)
    sgn = 1.0 if direction == 'max' else -1.0
    pv_clean = ctx.pv_official if official_pv_fn is not None else pp(x0)
    out0 = forward_all(adapter, ctx, pv_clean, need_grad=False)
    with torch.no_grad():
        clean_scores = {k: None if v is None else float(v) for k, v in bank.score_all(ctx, out0, item_id, want=tuple(ESTIMATORS)).items()}
        con0 = constraint_from_out(ctx, out0, tau_safe=cfg.tau_safe)
    clean_obj = clean_scores[objective]
    restarts = []
    pools = []
    for r in range(cfg.n_restarts):
        seed = restart_seed(item_id, objective, direction, r)
        if r == 0:
            x = x0.clone()
            n_shrink, start_feasible = (0, bool(con0['feasible']))
        else:
            x, n_shrink, start_feasible = _random_feasible_start(adapter, bank, ctx, pp, x0, cfg, seed, objective, item_id)
        pool = CandidatePool(cfg.keep_per_restart)
        pool_all = CandidatePool(cfg.keep_per_restart)
        alpha = cfg.alpha0
        traj = []
        best_harm = float('-inf')
        ckpt_ref = None
        alpha_events = []
        n_nonfinite_grad = 0
        m_t = torch.zeros_like(x0) if cfg.optimizer == 'adam' else None
        v_t = torch.zeros_like(x0) if cfg.optimizer == 'adam' else None
        t_adam = 0
        for step in range(cfg.steps + 1):
            x = x.detach().requires_grad_(True)
            loss, score, con = _eval_point(adapter, bank, ctx, pp, x, objective, item_id, cfg.tau_safe, True)
            s = float(score.detach())
            mm = float(con['min_margin'].detach())
            harm = sgn * s
            feasible = con['feasible']
            pool_all.offer(harm, step, s, mm, x)
            if feasible:
                pool.offer(harm, step, s, mm, x)
                best_harm = max(best_harm, harm)
            if step % cfg.traj_every == 0 or step == cfg.steps:
                traj.append({'step': step, 'score': round(s, 6), 'feasible': bool(feasible), 'min_margin': round(mm, 4), 'best_harm': None if best_harm == float('-inf') else round(best_harm, 6), 'linf_255': int((quantized_uint8(x).to(torch.int32) - quantized_uint8(x0).to(torch.int32)).abs().max()), 'alpha_255': round((cfg.adam_lr if cfg.optimizer == 'adam' else alpha) * 255.0, 4), 'ecdf': None if ecdf is None else round(ecdf(objective, s), 6)})
            if step == 0:
                ckpt_ref = best_harm
            if step == cfg.steps:
                break
            grad = torch.autograd.grad(sgn * loss, x)[0]
            finite = bool(torch.isfinite(grad).all())
            if not finite:
                n_nonfinite_grad += 1
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                if cfg.optimizer == 'adam':
                    assert cfg.adam_weight_decay == 0.0, 'the gate fixes Adam weight decay at 0'
                    t_adam += 1
                    b1, b2 = cfg.adam_betas
                    m_t = b1 * m_t + (1 - b1) * grad
                    v_t = b2 * v_t + (1 - b2) * grad * grad
                    mhat = m_t / (1 - b1 ** t_adam)
                    vhat = v_t / (1 - b2 ** t_adam)
                    x = project(x + cfg.adam_lr * mhat / (vhat.sqrt() + cfg.adam_eps), x0, cfg.eps).detach()
                else:
                    x = project(x + alpha * grad.sign(), x0, cfg.eps).detach()
            nxt = step + 1
            if cfg.optimizer == 'sign' and nxt in cfg.checkpoints:
                improved = best_harm - ckpt_ref if best_harm != float('-inf') and ckpt_ref is not None and (ckpt_ref != float('-inf')) else None if best_harm == float('-inf') else float('inf')
                if improved is None or improved < cfg.improve_tol:
                    new_alpha = max(alpha / 2.0, cfg.alpha_min)
                    halved = new_alpha < alpha
                    alpha = new_alpha
                else:
                    halved = False
                alpha_events.append({'checkpoint': nxt, 'improvement': None if improved is None or improved == float('inf') else round(improved, 8), 'alpha_255_after': round(alpha * 255, 4), 'halved': halved})
                ckpt_ref = best_harm
        restarts.append({'restart': r, 'seed': seed, 'start': 'clean' if r == 0 else 'random', 'random_start_shrinks': n_shrink, 'start_feasible': start_feasible, 'n_feasible_steps': sum((1 for t in traj if t['feasible'])), 'n_nonfinite_grad': n_nonfinite_grad, 'best_harm': None if best_harm == float('-inf') else round(best_harm, 6), 'alpha_events': alpha_events, 'traj': traj})
        pools.append((pool, pool_all))
    feas, infeas = ([], [])
    for r, (pool, pool_all) in enumerate(pools):
        seen = {c[1] for c in pool.items}
        for harm, step, sc, mm, u8 in pool.items:
            feas.append({'restart': r, 'step': step, 'harm': harm, 'score': sc, 'min_margin': mm, 'u8': u8, 'tf_feasible': True})
        for harm, step, sc, mm, u8 in pool_all.items:
            if step not in seen:
                infeas.append({'restart': r, 'step': step, 'harm': harm, 'score': sc, 'min_margin': mm, 'u8': u8, 'tf_feasible': False})
    feas.sort(key=lambda c: -c['harm'])
    infeas.sort(key=lambda c: -c['harm'])
    cands = feas + infeas
    chosen = None
    n_decoded = 0
    rejects = []
    for c in cands:
        raw = c['u8'].to(dev).to(torch.float32) / 255.0
        pv = official_pv_fn(c['u8']).to(dev) if official_pv_fn is not None else pp(raw)
        text, ids = fresh_decode(adapter, ctx, pv)
        n_decoded += 1
        if text == ctx.row['committed_clean_answer']:
            chosen = dict(c)
            chosen['decoded_answer'] = text
            chosen['decoded_ids'] = ids
            chosen['pv'] = pv
            break
        rejects.append({'restart': c['restart'], 'step': c['step'], 'harm': round(c['harm'], 6), 'decoded_answer': text})
    used_fallback = chosen is None
    if used_fallback:
        adv_u8 = quantized_uint8(x0).cpu()
        endpoint_scores = dict(clean_scores)
        endpoint_meta = None
        endpoint_source = 'clean_fallback'
        endpoint_feasible = bool(con0['feasible'])
        endpoint_margin = round(float(con0['min_margin']), 4)
    else:
        adv_u8 = chosen['u8']
        out_e = forward_all(adapter, ctx, chosen['pv'], need_grad=False)
        with torch.no_grad():
            endpoint_scores = {k: None if v is None else float(v) for k, v in bank.score_all(ctx, out_e, item_id, want=tuple(ESTIMATORS)).items()}
            con_e = constraint_from_out(ctx, out_e, tau_safe=cfg.tau_safe)
        endpoint_source = f"restart{chosen['restart']}_step{chosen['step']}"
        endpoint_feasible = bool(con_e['feasible'])
        endpoint_margin = round(float(con_e['min_margin']), 4)
        endpoint_meta = {'restart': chosen['restart'], 'step': chosen['step'], 'min_margin': round(chosen['min_margin'], 4), 'inloop_score': round(chosen['score'], 6), 'official_score': round(endpoint_scores[objective], 6), 'inloop_vs_official_abs': round(abs(chosen['score'] - endpoint_scores[objective]), 8), 'decoded_answer': chosen['decoded_answer'], 'decoded_ids': chosen['decoded_ids'], 'tf_feasible': bool(chosen['tf_feasible'])}
    return {'objective': objective, 'direction': direction, 'config': cfg.label, 'optimizer': cfg.optimizer, 'clean_scores': {k: None if v is None else round(v, 6) for k, v in clean_scores.items()}, 'endpoint_scores': {k: None if v is None else round(v, 6) for k, v in endpoint_scores.items()}, 'clean_objective': round(clean_obj, 6), 'endpoint_objective': round(endpoint_scores[objective], 6), 'raw_displacement': round(endpoint_scores[objective] - clean_obj, 6), 'clean_feasible': bool(con0['feasible']), 'clean_min_margin': round(float(con0['min_margin']), 4), 'endpoint_feasible': endpoint_feasible, 'endpoint_min_margin': endpoint_margin, 'used_clean_fallback': used_fallback, 'endpoint_source': endpoint_source, 'chosen': endpoint_meta, 'n_candidates': len(cands), 'n_candidates_tf_feasible': len(feas), 'n_candidates_tf_infeasible': len(infeas), 'n_fresh_decoded': n_decoded, 'n_fresh_rejected': len(rejects), 'fresh_rejects': rejects[:8], 'nontrivial_success': not used_fallback and chosen['step'] > 0, 'restarts': restarts, 'adv_u8': adv_u8}
