from __future__ import annotations
import hashlib
from dataclasses import asdict, dataclass, field
import torch
from src.inference.engine import fresh_decode, score
from src.inference.pil_exact import quantize_ste
EPS_DEFAULT = 8.0 / 255.0

@dataclass
class AttackConfig:
    eps: float = EPS_DEFAULT
    tau_safe: float = 0.0
    steps: int = 60
    n_restarts: int = 3
    alpha0: float = 2.0 / 255.0
    alpha_min: float = 0.25 / 255.0
    checkpoints: tuple = (15, 30, 45)
    improve_tol: float = 0.0001
    keep_per_restart: int = 5
    random_start_max_shrink: int = 4
    label: str = 'primary_60x3'

    def to_json(self):
        d = asdict(self)
        d['checkpoints'] = list(d['checkpoints'])
        return d

def restart_seed(item_id: str, direction: str, restart: int) -> int:
    h = hashlib.md5(f'{item_id}|{direction}|{restart}'.encode()).hexdigest()
    return 23 ^ int(h, 16) & 2147483647

def project(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(torch.min(torch.max(x, x0 - eps), x0 + eps), 0.0, 1.0)

def _objective(sc, direction: str) -> float:
    v = float(sc['tokprob'].detach()) if torch.is_tensor(sc['tokprob']) else float(sc['tokprob'])
    return -v if direction == 'min' else v

def quantized_uint8(raw01: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(raw01.detach() * 255.0), 0, 255).to(torch.uint8)

class CandidatePool:

    def __init__(self, k: int):
        self.k = k
        self.items = []

    def offer(self, harm, step, tokprob, min_margin, raw01):
        if len(self.items) >= self.k and harm <= self.items[-1][0]:
            return
        self.items.append((harm, step, tokprob, min_margin, quantized_uint8(raw01).cpu()))
        self.items.sort(key=lambda t: -t[0])
        del self.items[self.k:]

def _random_feasible_start(adapter, ctx, pp, x0, cfg, seed, direction):
    g = torch.Generator(device=x0.device)
    g.manual_seed(seed)
    d = (torch.rand(x0.shape, generator=g, device=x0.device, dtype=x0.dtype) * 2.0 - 1.0) * cfg.eps
    for shrink in range(cfg.random_start_max_shrink + 1):
        x = project(x0 + d / 2 ** shrink, x0, cfg.eps)
        sc = score(adapter, ctx, pp(x), tau_safe=cfg.tau_safe)
        if sc['feasible']:
            return (x.detach(), shrink, True)
    return (x0.clone(), cfg.random_start_max_shrink + 1, False)

def run_direction(adapter, pp, ctx, direction: str, cfg: AttackConfig, official_pv_fn=None) -> dict:
    dev = ctx.raw01.device
    x0 = ctx.raw01.detach().to(torch.float32)
    sc0 = score(adapter, ctx, ctx.pv_official if official_pv_fn is not None else pp(x0), tau_safe=cfg.tau_safe)
    clean_tokprob = float(sc0['tokprob'])
    restarts = []
    pools = []
    for r in range(cfg.n_restarts):
        seed = restart_seed(ctx.row['item_id'], direction, r)
        if r == 0:
            x = x0.clone()
            n_shrink, start_feasible = (0, bool(sc0['feasible']))
        else:
            x, n_shrink, start_feasible = _random_feasible_start(adapter, ctx, pp, x0, cfg, seed, direction)
        pool = CandidatePool(cfg.keep_per_restart)
        pool_all = CandidatePool(cfg.keep_per_restart)
        alpha = cfg.alpha0
        traj = []
        best_harm = float('-inf')
        ckpt_ref = None
        alpha_events = []
        for step in range(cfg.steps + 1):
            x = x.detach().requires_grad_(True)
            sc = score(adapter, ctx, pp(x), tau_safe=cfg.tau_safe, need_grad=True)
            harm = _objective(sc, direction)
            feasible = sc['feasible']
            pool_all.offer(harm, step, float(sc['tokprob']), float(sc['min_margin']), x)
            if feasible:
                pool.offer(harm, step, float(sc['tokprob']), float(sc['min_margin']), x)
                best_harm = max(best_harm, harm)
            traj.append({'step': step, 'tokprob': round(float(sc['tokprob']), 6), 'feasible': bool(feasible), 'min_margin': round(float(sc['min_margin']), 4), 'best_harm': None if best_harm == float('-inf') else round(best_harm, 6), 'alpha_255': round(alpha * 255.0, 4)})
            if step == 0:
                ckpt_ref = best_harm
            if step == cfg.steps:
                break
            loss = sc['span_logsum'] if direction == 'max' else -sc['span_logsum']
            grad = torch.autograd.grad(loss, x)[0]
            with torch.no_grad():
                x = project(x + alpha * grad.sign(), x0, cfg.eps).detach()
            nxt = step + 1
            if nxt in cfg.checkpoints:
                improved = best_harm - ckpt_ref if best_harm != float('-inf') and ckpt_ref is not None and (ckpt_ref != float('-inf')) else None if best_harm == float('-inf') else float('inf')
                if improved is None or improved < cfg.improve_tol:
                    new_alpha = max(alpha / 2.0, cfg.alpha_min)
                    alpha_events.append({'checkpoint': nxt, 'improvement': None if improved is None else None if improved == float('inf') else round(improved, 8), 'alpha_255_before': round(alpha * 255, 4), 'alpha_255_after': round(new_alpha * 255, 4), 'halved': new_alpha < alpha})
                    alpha = new_alpha
                else:
                    alpha_events.append({'checkpoint': nxt, 'improvement': round(improved, 8), 'alpha_255_before': round(alpha * 255, 4), 'alpha_255_after': round(alpha * 255, 4), 'halved': False})
                ckpt_ref = best_harm
        restarts.append({'restart': r, 'seed': seed, 'start': 'clean' if r == 0 else 'random', 'random_start_shrinks': n_shrink, 'start_feasible': start_feasible, 'n_feasible_steps': sum((1 for t in traj if t['feasible'])), 'best_harm': None if best_harm == float('-inf') else round(best_harm, 6), 'alpha_events': alpha_events, 'traj': traj})
        pools.append((pool, pool_all))
    feas, infeas = ([], [])
    for r, (pool, pool_all) in enumerate(pools):
        seen = {c[1] for c in pool.items}
        for harm, step, tp, mm, u8 in pool.items:
            feas.append({'restart': r, 'step': step, 'harm': harm, 'tokprob': tp, 'min_margin': mm, 'u8': u8, 'tf_feasible': True})
        for harm, step, tp, mm, u8 in pool_all.items:
            if step not in seen:
                infeas.append({'restart': r, 'step': step, 'harm': harm, 'tokprob': tp, 'min_margin': mm, 'u8': u8, 'tf_feasible': False})
    feas.sort(key=lambda c: -c['harm'])
    infeas.sort(key=lambda c: -c['harm'])
    cands = feas + infeas
    chosen = None
    n_decoded = 0
    rejects = []
    for c in cands:
        raw = c['u8'].to(dev).to(torch.float32) / 255.0
        if official_pv_fn is not None:
            pv = official_pv_fn(c['u8']).to(dev)
        else:
            pv = pp(raw)
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
        endpoint_tokprob = clean_tokprob
        endpoint_source = 'clean_fallback'
        adv_u8 = quantized_uint8(x0).cpu()
        chosen_meta = None
        official = {'tokprob': clean_tokprob, 'feasible': bool(sc0['feasible']), 'min_margin': round(float(sc0['min_margin']), 4)}
    else:
        adv_u8 = chosen['u8']
        sc_off = score(adapter, ctx, chosen['pv'], tau_safe=cfg.tau_safe)
        official = {'tokprob': round(float(sc_off['tokprob']), 6), 'feasible': bool(sc_off['feasible']), 'min_margin': round(float(sc_off['min_margin']), 4)}
        endpoint_tokprob = official['tokprob']
        endpoint_source = f"restart{chosen['restart']}_step{chosen['step']}"
        chosen_meta = {'restart': chosen['restart'], 'step': chosen['step'], 'min_margin': round(chosen['min_margin'], 4), 'inloop_tokprob': round(chosen['tokprob'], 6), 'official_tokprob': official['tokprob'], 'inloop_vs_official_abs': round(abs(chosen['tokprob'] - official['tokprob']), 8), 'decoded_answer': chosen['decoded_answer'], 'decoded_ids': chosen['decoded_ids']}
    return {'direction': direction, 'clean_tokprob': round(clean_tokprob, 6), 'clean_feasible': bool(sc0['feasible']), 'clean_min_margin': round(float(sc0['min_margin']), 4), 'endpoint_tokprob': round(endpoint_tokprob, 6), 'endpoint_official_feasible': official['feasible'], 'endpoint_official_min_margin': official['min_margin'], 'displacement': round(endpoint_tokprob - clean_tokprob, 6), 'used_clean_fallback': used_fallback, 'endpoint_source': endpoint_source, 'chosen': chosen_meta, 'n_candidates': len(cands), 'n_candidates_tf_feasible': len(feas), 'n_candidates_tf_infeasible': len(infeas), 'accepted_tf_feasible': None if used_fallback else bool(chosen['tf_feasible']), 'n_fresh_decoded': n_decoded, 'n_fresh_rejected': len(rejects), 'fresh_rejects': rejects[:8], 'nontrivial_success': not used_fallback and chosen['step'] > 0, 'restarts': restarts, 'adv_u8': adv_u8}

def validate_candidate(adv_u8: torch.Tensor, clean_u8: torch.Tensor, eps: float, pp, official_pv_fn, tol: float=0.0) -> dict:
    adv01 = adv_u8.to(torch.float32) / 255.0
    gl = int((adv_u8.to(torch.int32) - clean_u8.to(torch.int32)).abs().max())
    eps_gl = eps * 255.0
    out = {'min_pixel': float(adv01.min()), 'max_pixel': float(adv01.max()), 'linf_raw': gl / 255.0, 'linf_greylevels': gl, 'eps_greylevels': eps_gl, 'within_eps': bool(gl <= eps_gl + 1e-09), 'in_unit_box': bool(adv01.min() >= 0.0 and adv01.max() <= 1.0), 'finite': bool(torch.isfinite(adv01).all()), 'endpoint_is_clean_image': bool(gl == 0), 'endpoint_u8_sha256': hashlib.sha256(adv_u8.detach().cpu().contiguous().numpy().tobytes()).hexdigest(), 'clean_u8_sha256': hashlib.sha256(clean_u8.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
    if official_pv_fn is not None:
        pv_off = official_pv_fn(adv_u8)
        pv_diff = pp(adv01.to(pv_off.device if hasattr(pv_off, 'device') else 'cpu').double())
        out['pv_shape_match'] = tuple(pv_off.shape) == tuple(pv_diff.shape)
        if out['pv_shape_match']:
            d = (pv_off.float() - pv_diff.float().to(pv_off.device)).abs()
            out['pv_max_abs_diff'] = float(d.max())
            out['pv_exact'] = bool(out['pv_max_abs_diff'] <= max(tol, 1e-05))
    return out
