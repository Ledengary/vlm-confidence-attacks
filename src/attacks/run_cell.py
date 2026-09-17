from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.config import DATA_DIR, SEED
from src.attacks.kl_gate import kl_enforced, realized_kl, stash_clean
from src.attacks import gate_v12
OUT = DATA_DIR / 'kl_constraint'
CONFIG = 'B_corrected_30x2'
READOUTS = ('tokprob', 'pik', 'saplma', 'ccpsd', 'ccps', 'full', 'coupled')
ATTACKED = ('pik', 'saplma')
DIRECTIONS = ('min', 'max')

def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = (y[m], s[m])
    if len(set(y.tolist())) < 2:
        return float('nan')
    o = np.argsort(s, kind='mergesort')
    rk = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[o[j + 1]] == s[o[i]]:
            j += 1
        rk[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def balanced_items(model, setname, n, seed=SEED):
    from src.data_prep.canonical import load_manifest
    rows = [r for r in load_manifest(model) if r['dataset'] == setname]
    corr = [r for r in rows if int(r['label']) == 1]
    wrong = [r for r in rows if int(r['label']) == 0]
    rng = np.random.default_rng(seed)
    k = min(len(corr), len(wrong), n // 2)
    cs = [corr[i] for i in rng.permutation(len(corr))[:k]]
    ws = [wrong[i] for i in rng.permutation(len(wrong))[:k]]
    sub = cs + ws
    sub.sort(key=lambda r: r['item_id'])
    return (sub, k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--set', dest='setname', required=True)
    ap.add_argument('--celltag', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--deltas', default='0.2,0.05')
    ap.add_argument('--deadline', type=float, default=0.0, help='epoch seconds; stop starting items after')
    a = ap.parse_args()
    from src.data_prep.canonical import load_manifest
    from src.inference.engine import build_item, load_adapter
    from src.attacks.official_pv import official_pv_fn_factory
    from src.estimators.objectives import forward_all
    from src.attacks.strict_bank import StrictCanonicalBank
    deltas = [float(x) for x in a.deltas.split(',')]
    dev = f'cuda:{a.gpu}'
    cell_dir = OUT / a.celltag
    cell_dir.mkdir(parents=True, exist_ok=True)
    rec_path = cell_dir / 'records.jsonl'
    done_ids = set()
    if rec_path.exists():
        for l in open(rec_path):
            if l.strip():
                try:
                    done_ids.add(json.loads(l)['item_id'])
                except Exception:
                    pass
    items, k = balanced_items(a.model, a.setname, a.n)
    adapter, pp, _ = load_adapter(a.model, a.gpu)
    bank = StrictCanonicalBank(a.model, adapter, dev)
    manifest = {r['item_id']: r for r in load_manifest(a.model)}
    fh = open(rec_path, 'a')
    t0 = time.time()
    n_ok = n_err = 0
    for it, row in enumerate(items):
        if row['item_id'] in done_ids:
            continue
        if a.deadline and time.time() > a.deadline:
            print(f'[{a.celltag}] deadline reached at item {it}; stopping', flush=True)
            break
        try:
            ctx = build_item(adapter, pp, manifest[row['item_id']], dev)
            off = official_pv_fn_factory(adapter, manifest[row['item_id']])
            out0 = forward_all(adapter, ctx, ctx.pv_official, need_grad=False)
            stash_clean(ctx, out0)
            with torch.no_grad():
                clean_scores = {kk: None if v is None else float(v) for kk, v in bank.score_all(ctx, out0, row['item_id'], want=READOUTS).items()}
            rec = {'item_id': row['item_id'], 'label': int(row['label']), 'clean_scores': clean_scores, 'endpoints': {}}
            for delta in deltas:
                dtag = 'inf' if delta == float('inf') else str(delta)
                with kl_enforced(adapter, delta):
                    for obj in ATTACKED:
                        for direction in DIRECTIONS:
                            ctx._kl_accepted = None
                            r = gate_v12.run_one(adapter, bank, pp, ctx, row['item_id'], a.model, obj, direction, CONFIG, off)
                            if r['used_clean_fallback']:
                                rk = {'kl_max': 0.0, 'kl_mean': 0.0}
                            else:
                                rk = ctx._kl_accepted or {'kl_max': None, 'kl_mean': None}
                            rec['endpoints'][f'{dtag}|{obj}|{direction}'] = {'endpoint_scores': r['endpoint_scores'], 'used_clean_fallback': bool(r['used_clean_fallback']), 'n_feasible_steps': sum((m['n_feasible_steps'] for m in r['restarts'])), 'kl_max': rk['kl_max'], 'kl_mean': rk['kl_mean']}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            n_ok += 1
            del ctx, out0
        except Exception as e:
            import traceback
            fh.write(json.dumps({'item_id': row['item_id'], 'ERROR': str(e)[:300], 'trace': traceback.format_exc()[-400:]}) + '\n')
            fh.flush()
            n_err += 1
        if (n_ok + n_err) % 10 == 0:
            el = time.time() - t0
            print(f'[{a.celltag}] {n_ok + n_err} done ({n_err} err) {el / max(n_ok + n_err, 1):.1f}s/item', flush=True)
    fh.close()
    analyze(a.model, a.setname, a.celltag, deltas, k)
    print(f'[{a.celltag}] DONE ok={n_ok} err={n_err} {time.time() - t0:.0f}s', flush=True)

def analyze(model, setname, celltag, deltas, k):
    cell_dir = OUT / celltag
    recs = []
    n_err = 0
    for l in open(cell_dir / 'records.jsonl'):
        if l.strip():
            r = json.loads(l)
            if 'ERROR' in r:
                n_err += 1
            else:
                recs.append(r)
    if not recs:
        json.dump({'cell': f'{model}/{setname}', 'n': 0, 'n_err': n_err, 'note': 'no records'}, open(cell_dir / 'result.json', 'w'), indent=2)
        return
    y = np.array([r['label'] for r in recs])
    dtags = ['inf' if d == float('inf') else str(d) for d in deltas]
    res = {'cell': f'{model}/{setname}', 'celltag': celltag, 'n': len(recs), 'n_correct': int((y == 1).sum()), 'n_wrong': int((y == 0).sum()), 'n_err': n_err, 'deltas': dtags, 'readouts': {}, 'kl_verify': {}, 'feasibility': {}}
    clean_au = {}
    for rd in READOUTS:
        s = np.array([r['clean_scores'].get(rd) for r in recs], float)
        clean_au[rd] = round(auroc(y, s), 4) if np.isfinite(s).any() else None
    res['clean_auroc'] = clean_au
    for dtag in dtags:
        res['readouts'][dtag] = {}
        for rd in ATTACKED:
            s_adv = []
            for r in recs:
                cand = [r['clean_scores'].get(rd)]
                for d in DIRECTIONS:
                    ep = r['endpoints'].get(f'{dtag}|{rd}|{d}')
                    if ep:
                        cand.append(ep['endpoint_scores'].get(rd))
                cand = [c for c in cand if c is not None]
                if not cand:
                    s_adv.append(np.nan)
                    continue
                s_adv.append(min(cand) if r['label'] == 1 else max(cand))
            aa = auroc(y, np.array(s_adv, float))
            res['readouts'][dtag][rd] = {'attacked_auroc': round(aa, 4), 'below_chance': bool(aa < 0.5), 'clean_auroc': clean_au[rd], 'type': 'direct'}
        for rd in READOUTS:
            if rd in ATTACKED:
                continue
            s_adv = []
            for r in recs:
                cand = [r['clean_scores'].get(rd)]
                for key, ep in r['endpoints'].items():
                    if key.startswith(dtag + '|'):
                        cand.append(ep['endpoint_scores'].get(rd))
                cand = [c for c in cand if c is not None]
                if not cand:
                    s_adv.append(np.nan)
                    continue
                s_adv.append(min(cand) if r['label'] == 1 else max(cand))
            aa = auroc(y, np.array(s_adv, float))
            res['readouts'][dtag][rd] = {'attacked_auroc': round(aa, 4), 'below_chance': bool(aa < 0.5), 'clean_auroc': clean_au[rd], 'type': 'transfer'}
        klmax, klmean, feas, nfb, ntot = ([], [], [], 0, 0)
        for r in recs:
            for key, ep in r['endpoints'].items():
                if not key.startswith(dtag + '|'):
                    continue
                ntot += 1
                if ep['used_clean_fallback']:
                    nfb += 1
                else:
                    klmax.append(ep['kl_max'])
                    klmean.append(ep['kl_mean'])
                feas.append(ep['n_feasible_steps'] / 62.0)
        res['kl_verify'][dtag] = {'realized_kl_max_over_endpoints': round(float(np.max(klmax)), 6) if klmax else None, 'realized_kl_mean_over_endpoints': round(float(np.mean(klmax)), 6) if klmax else None, 'n_nonclean_endpoints': len(klmax), 'n_clean_fallback': nfb, 'n_endpoints': ntot, 'clean_fallback_rate': round(nfb / max(ntot, 1), 4)}
        res['feasibility'][dtag] = {'mean_kl_feasible_step_rate': round(float(np.mean(feas)), 4) if feas else None, 'median': round(float(np.median(feas)), 4) if feas else None}
    json.dump(res, open(cell_dir / 'result.json', 'w'), indent=2, default=str)
if __name__ == '__main__':
    main()
