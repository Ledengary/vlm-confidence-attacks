from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
from PIL import Image
from src.config import DATA_DIR
from src.attacks.attack import project, quantized_uint8, validate_candidate
from src.data_prep.canonical import load_manifest
from src.inference.engine import build_item, fresh_decode, load_adapter
from src.attacks.official_pv import official_pv_fn_factory
from src.attacks.panel import load_panel
from src.estimators.readout_bank import ESTIMATORS
from src.attacks.gate_attack import restart_seed
from src.estimators.objectives import constraint_from_out, forward_all, objective_logit
from src.attacks.selection import KEEP, WINDOWS, TwoTierWindowedPools, assert_no_better_later, containment_audit, order_and_select
from src.attacks.store import ShardWriter, scan
from src.attacks.canonical_bank import CanonicalBank
V12 = DATA_DIR / 'snowglobe' / 'pfmc_v1_2'
V11 = DATA_DIR / 'snowglobe' / 'pfmc_v1_1'
EPS = 8.0 / 255.0
ALPHA0, ALPHA_MIN = (2.0 / 255.0, 0.25 / 255.0)
CHECKPOINTS = (15, 30, 45)
SPSA_C = 1.0 / 255.0
CONFIGS = {'C_corrected_60x3': {'steps': 60, 'restarts': 3}, 'B_corrected_30x2': {'steps': 30, 'restarts': 2}}

def _warm_from_corrected_c(model, item_id, direction):
    for f in sorted((V11 / 'corrected_c_v11').glob(f'{model}_s*.jsonl')):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if 'ERROR' in r or r['item_id'] != item_id or r['objective'] != 'ccpsd' or (r['direction'] != direction) or r['used_clean_fallback']:
                continue
            p = DATA_DIR / 'snowglobe' / 'all_channel_gate' / 'endpoints' / model / f'{item_id}__ccpsd__{direction}__C_60x3.png'
            if p.exists():
                return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2, 0, 1)
    return None

def run_one(adapter, bank, pp, ctx, item_id, model, objective, direction, cfg_name, off) -> dict:
    spec = CONFIGS[cfg_name]
    dev = ctx.raw01.device
    x0 = ctx.raw01.detach().to(torch.float32)
    sgn = 1.0 if direction == 'max' else -1.0
    clean_u8 = quantized_uint8(x0).cpu()
    is_direct = objective == 'ccps_direct'
    method = 'spsa' if is_direct else 'autograd'

    def score_of(out, need_grad):
        if is_direct:
            return bank.logit_and_score('ccps', ctx, out)
        if objective in ('ccps', 'ccpsd'):
            return bank.logit_and_score(objective, ctx, out)
        return objective_logit(bank, ctx, out, item_id, objective)
    out0 = forward_all(adapter, ctx, ctx.pv_official, need_grad=False)
    with torch.no_grad():
        clean_scores = {k: None if v is None else float(v) for k, v in bank.score_all(ctx, out0, item_id, want=tuple(ESTIMATORS)).items()}
        _, s0 = score_of(out0, False)
        con0 = constraint_from_out(ctx, out0, tau_safe=0.0)
    clean = float(s0)
    kinds = ['clean', 'random']
    if spec['restarts'] >= 3:
        kinds.append('ccpsd_warm' if is_direct else 'random2')
    pools_by_restart, trajectories, meta = ([], [], [])
    for r, kind in enumerate(kinds[:spec['restarts']]):
        seed = restart_seed(item_id, objective, direction, r)
        if kind == 'clean':
            x = x0.clone()
        elif kind == 'ccpsd_warm':
            w = _warm_from_corrected_c(model, item_id, direction)
            x = project(w.to(dev).to(torch.float32) / 255.0, x0, EPS) if w is not None else x0.clone()
        else:
            g = torch.Generator(device=dev)
            g.manual_seed(seed)
            d = (torch.rand(x0.shape, generator=g, device=dev, dtype=x0.dtype) * 2 - 1) * EPS
            x = x0.clone()
            for s in range(5):
                xt = project(x0 + d / 2 ** s, x0, EPS)
                o = forward_all(adapter, ctx, pp(xt), need_grad=False)
                if constraint_from_out(ctx, o, tau_safe=0.0)['feasible']:
                    x = xt.detach()
                    break
        pools = TwoTierWindowedPools(KEEP, WINDOWS)
        traj = []
        alpha, best, ck = (ALPHA0, float('-inf'), None)
        gs = torch.Generator(device=dev)
        gs.manual_seed(seed)
        for step in range(spec['steps'] + 1):
            if method == 'autograd':
                x = x.detach().requires_grad_(True)
                out = forward_all(adapter, ctx, pp(x), need_grad=True)
                loss, sc = score_of(out, True)
                con = constraint_from_out(ctx, out, tau_safe=0.0)
                s, mm = (float(sc.detach()), float(con['min_margin'].detach()))
            else:
                out = forward_all(adapter, ctx, pp(x), need_grad=False)
                with torch.no_grad():
                    _, sc = score_of(out, False)
                    con = constraint_from_out(ctx, out, tau_safe=0.0)
                s, mm = (float(sc), float(con['min_margin']))
            harm = sgn * s
            pools.offer(harm, step, s, mm, x, bool(con['feasible']))
            traj.append({'harm': harm, 'step': step, 'score': s, 'min_margin': mm, 'u8': quantized_uint8(x), 'feasible': bool(con['feasible'])})
            if con['feasible']:
                best = max(best, harm)
            if step == 0:
                ck = best
            if step == spec['steps']:
                break
            if method == 'autograd':
                gr = torch.autograd.grad(sgn * loss, x)[0]
                upd = alpha * torch.nan_to_num(gr, nan=0.0, posinf=0.0, neginf=0.0).sign()
            else:
                delta = torch.randint(0, 2, x0.shape, generator=gs, device=dev, dtype=torch.float32) * 2 - 1
                sp = _probe(adapter, ctx, pp, project(x + SPSA_C * delta, x0, EPS), score_of)
                sm = _probe(adapter, ctx, pp, project(x - SPSA_C * delta, x0, EPS), score_of)
                upd = alpha * (sgn * (sp - sm) / (2 * SPSA_C) * delta).sign()
            with torch.no_grad():
                x = project(x + upd, x0, EPS).detach()
            if step + 1 in CHECKPOINTS:
                imp = best - ck if best != float('-inf') and ck not in (None, float('-inf')) else None
                if imp is None or imp < 0.0001:
                    alpha = max(alpha / 2.0, ALPHA_MIN)
                ck = best
        pools_by_restart.append(pools)
        trajectories.append(traj)
        meta.append({'restart': r, 'start': kind, 'seed': seed, 'n_feasible_steps': sum((1 for t in traj if t['feasible']))})
    cands = []
    for r, p in enumerate(pools_by_restart):
        cands.extend(p.union(r).values())
    n_before = sum((len(p.all[w]) + len(p.feas[w]) for p in pools_by_restart for w in WINDOWS))

    def official(u8):
        pv = off(u8).to(dev)
        o = forward_all(adapter, ctx, pv, need_grad=False)
        with torch.no_grad():
            _, s = score_of(o, False)
            c = constraint_from_out(ctx, o, tau_safe=0.0)
        return (float(s), bool(c['feasible']), round(float(c['min_margin']), 4), pv)
    committed = ctx.row['committed_clean_answer']
    chosen, ordered, n_off, n_dec, rejects = order_and_select(cands, sgn, clean_u8, official, lambda pv: fresh_decode(adapter, ctx, pv), committed)
    nb = assert_no_better_later(chosen, ordered, lambda pv: fresh_decode(adapter, ctx, pv), committed, check_all=True)
    cont = containment_audit(model, item_id, objective, direction, pools_by_restart, trajectories)
    is_clean = bool(chosen.get('is_clean'))
    if is_clean:
        endpoint_scores, endpoint = (dict(clean_scores), clean)
    else:
        oe = forward_all(adapter, ctx, chosen['pv'], need_grad=False)
        with torch.no_grad():
            endpoint_scores = {k: None if v is None else float(v) for k, v in bank.score_all(ctx, oe, item_id, want=tuple(ESTIMATORS)).items()}
        endpoint = chosen['official_score']
    return {'model': model, 'item_id': item_id, 'dataset': ctx.row['dataset'], 'objective': objective, 'direction': direction, 'config': cfg_name, 'method': method, 'clean_scores': {k: None if v is None else round(v, 6) for k, v in clean_scores.items()}, 'endpoint_scores': {k: None if v is None else round(v, 6) for k, v in endpoint_scores.items()}, 'clean_objective': round(clean, 6), 'endpoint_objective': round(endpoint, 6), 'endpoint_source': 'clean_guaranteed' if is_clean else f"restart{chosen['restart']}_step{chosen['step']}", 'used_clean_fallback': is_clean, 'accepted_start_kind': None if is_clean else meta[chosen['restart']]['start'], 'accepted_step': None if is_clean else chosen['step'], 'accepted_restart': None if is_clean else chosen['restart'], 'endpoint_feasible': chosen['official_feasible'], 'endpoint_min_margin': chosen['official_min_margin'], 'clean_feasible': bool(con0['feasible']), 'n_candidates_before_dedup': n_before, 'n_candidates_after_dedup': len(cands), 'n_official_rescoring_forwards': n_off, 'n_fresh_decoded': n_dec, 'n_fresh_rejected': len(rejects), 'no_better_later': nb, 'containment': cont, 'validity': validate_candidate(chosen['u8'], clean_u8, EPS, pp, off), 'restarts': meta, 'canonical_bank': bank.canonical}

def _probe(adapter, ctx, pp, x, score_of):
    o = forward_all(adapter, ctx, pp(x), need_grad=False)
    with torch.no_grad():
        _, s = score_of(o, False)
    return float(s)

def run(model, gpu, objectives, config, shard, num_shards, out_name):
    out_dir = V12 / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = f'cuda:{gpu}'
    adapter, pp, pp_meta = load_adapter(model, gpu)
    bank = CanonicalBank(model, adapter, dev)
    manifest = {r['item_id']: r for r in load_manifest(model)}
    panel = [p for p in load_panel() if p['model'] == model]
    jobs = [(p, o, d) for p in panel for o in objectives for d in ('min', 'max')][shard::num_shards]
    res = scan(sorted(out_dir.glob(f'{model}_s*.jsonl')))
    todo = [(p, o, d) for p, o, d in jobs if (model, p['item_id'], o, d, config) not in res.done]
    print(f'[{out_name} {model} s{shard}] jobs={len(jobs)} todo={len(todo)} bank={bank.canonical}', flush=True)
    if not todo:
        return
    w = ShardWriter(out_dir / f'{model}_s{shard}.jsonl')
    t0 = time.time()
    for i, (p, obj, direction) in enumerate(todo):
        ti = time.time()
        try:
            row = manifest[p['item_id']]
            ctx = build_item(adapter, pp, row, dev)
            off = official_pv_fn_factory(adapter, row)
            rec = run_one(adapter, bank, pp, ctx, p['item_id'], model, obj, direction, config, off)
            rec['label'] = p['label']
            rec['group'] = p['group']
            rec['seconds'] = round(time.time() - ti, 3)
            rec['peak_mem_gb'] = round(torch.cuda.max_memory_allocated(gpu) / 2 ** 30, 3)
            w.write(rec)
        except Exception as e:
            import traceback
            w.write({'model': model, 'item_id': p['item_id'], 'objective': obj, 'direction': direction, 'config': config, 'ERROR': str(e)[:400], 'trace': traceback.format_exc()[-600:]})
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f'[{out_name} {model} s{shard}] {i + 1}/{len(todo)} {el / (i + 1):.1f}s/job eta={(len(todo) - i - 1) * el / (i + 1) / 60:.1f}min', flush=True)
    w.close()
    print(f'[{out_name} {model} s{shard}] DONE {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--objectives', required=True)
    ap.add_argument('--config', default='B_corrected_30x2', choices=list(CONFIGS))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run(a.model, a.gpu, a.objectives.split(','), a.config, a.shard, a.num_shards, a.out)
