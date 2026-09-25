from __future__ import annotations
import glob
import json
from pathlib import Path
import numpy as np
from src.config import DATA_DIR, SEED
from src.verifier_ladder.data import auroc
from src.verifier_ladder.span12 import MSHORT, SPAN12, SSHORT
OUT = DATA_DIR / 'snowglobe' / 'verifier_escape'
IV_VER = 'internvl3_5_2b'
SPSA_BUDGET = '600'
HSJ_BUDGET = '1000'
N_BOOT = 2000

def _load(pat):
    rows = []
    for f in sorted(glob.glob(str(OUT / pat))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if 'ERROR' not in r and 'item_id' in r:
                    rows.append(r)
    return rows

def _oracle(y, cols):
    cols = np.array(cols)
    s = np.where(np.array(y) == 1, cols.min(axis=0), cols.max(axis=0))
    return s

def _boot_ci(y, s, n=N_BOOT):
    y = np.asarray(y)
    s = np.asarray(s)
    rng = np.random.default_rng(SEED)
    m = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        yi, si = (y[idx], s[idx])
        if len(set(yi.tolist())) < 2:
            continue
        vals.append(auroc(yi, si))
    if not vals:
        return (None, None)
    return (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))

def _spsa_cap_ids(base, setname):
    p = OUT / f'sp_pool100_{MSHORT[base]}_{SSHORT[setname]}.json'
    return set(json.load(open(p))['item_ids']) if p.exists() else None

def rung_c0_transfer(sp):
    rows = _load(f'{sp}_spscores_s*.jsonl')
    if not rows:
        return None
    by = {r['item_id']: r for r in rows}
    y = [r['label'] for r in by.values()]
    c0 = [r['c0_clean'] for r in by.values()]
    tr = _oracle(y, [[r['c0_clean'] for r in by.values()], [r['c1_min'] for r in by.values()], [r['c1_max'] for r in by.values()]])
    return {'n': len(y), 'C0': {'auroc': round(auroc(y, c0), 4), 'ci': _boot_ci(y, c0)}, 'transfer': {'auroc': round(auroc(y, tr), 4), 'ci': _boot_ci(y, tr)}}

def rung_seeded_score(sp, kind, ids_keep=None):
    budget = HSJ_BUDGET if kind == 'hsj' else SPSA_BUDGET
    fpat = 'hardlabel' if kind == 'hsj' else 'query'
    per_seed = {}
    for sd in (1, 2):
        rows = _load(f'{sp}_{fpat}_seed{sd}_s*.jsonl')
        if ids_keep is not None:
            rows = [r for r in rows if r['item_id'] in ids_keep]
        if not rows:
            continue
        by = {r['item_id']: r for r in rows}
        y = [r['label'] for r in by.values()]
        s = [r['checkpoint_scores'].get(budget, list(r['checkpoint_scores'].values())[-1]) for r in by.values()]
        per_seed[sd] = {'n': len(y), 'auroc': round(auroc(y, s), 4), 'ci': _boot_ci(y, s)}
    return per_seed

def rung_c2(sp, ids_keep=None):
    per_seed = {}
    for sd in (1, 2):
        rows = _load(f'c2_{sp}_seed{sd}_scores_s*.jsonl')
        rows = [r for r in rows if 'direction' in r and 'verifier_score' in r]
        if ids_keep is not None:
            rows = [r for r in rows if r['item_id'] in ids_keep]
        if not rows:
            continue
        by = {}
        for r in rows:
            by.setdefault(r['item_id'], {})[r['direction']] = r['verifier_score']
        lab = {r['item_id']: r['label'] for r in rows}
        items = [i for i in by if len(by[i]) >= 1]
        y = [lab[i] for i in items]
        s = _oracle(y, [[min(by[i].values()) for i in items], [max(by[i].values()) for i in items]])
        per_seed[sd] = {'n': len(y), 'auroc': round(auroc(y, s), 4), 'ci': _boot_ci(y, s)}
    return per_seed

def _across_seed(per_seed):
    if not per_seed:
        return None
    aur = [v['auroc'] for v in per_seed.values()]
    return {'per_seed': {str(k): v for k, v in per_seed.items()}, 'mean': round(float(np.mean(aur)), 4), 'std': round(float(np.std(aur)), 4) if len(aur) > 1 else 0.0, 'n_seeds': len(aur)}

def c2_parity(sp):
    mx = []
    for f in sorted(glob.glob(str(OUT / f'c2_{sp}_seed*_parity_s*.jsonl'))):
        for line in open(f):
            if line.strip():
                mx.append(json.loads(line).get('abs_diff', 0.0))
    return round(max(mx), 8) if mx else None

def analyze_triple(sp, lab, base, ver, setname):
    iv = ver == IV_VER
    ids_keep = _spsa_cap_ids(base, setname) if iv else None
    row = {'triple': sp, 'label': lab, 'base': MSHORT[base], 'verifier': MSHORT[ver], 'dataset': SSHORT[setname], 'iv_verifier': iv, 'spsa_n': 100 if iv else 200}
    ct = rung_c0_transfer(sp)
    row['c0_transfer'] = ct
    row['hsj'] = _across_seed(rung_seeded_score(sp, 'hsj'))
    row['spsa'] = _across_seed(rung_seeded_score(sp, 'spsa', ids_keep=ids_keep))
    row['c2'] = _across_seed(rung_c2(sp))
    row['c2_parity_max_abs'] = c2_parity(sp)
    chain = []
    if ct:
        chain = [('C0', ct['C0']['auroc']), ('transfer', ct['transfer']['auroc'])]
    for k, key in (('decision', 'hsj'), ('score', 'spsa'), ('white-box', 'c2')):
        if row[key]:
            chain.append((k, row[key]['mean']))
    row['chain'] = chain
    viol = [(chain[i][0], chain[i + 1][0], round(chain[i + 1][1] - chain[i][1], 4)) for i in range(len(chain) - 1) if chain[i + 1][1] > chain[i][1] + 1e-09]
    row['monotonicity_violations'] = viol
    return row

def main():
    results = []
    for sp, lab, base, ver, setname in SPAN12:
        results.append(analyze_triple(sp, lab, base, ver, setname))
    complete = [r for r in results if r['c0_transfer'] and r['hsj'] and r['spsa'] and r['c2']]
    all_viol = sum((len(r['monotonicity_violations']) for r in complete))
    out = {'n_triples': len(results), 'n_complete': len(complete), 'monotonicity_violations_total': all_viol, 'c2_parity_max_over_triples': max([r['c2_parity_max_abs'] for r in results if r['c2_parity_max_abs'] is not None], default=None), 'triples': results}
    outp = OUT / 'span12_ladder_results.json'
    json.dump(out, open(outp, 'w'), indent=2, default=str)
    print(json.dumps({'n_complete': len(complete), 'monotonicity_violations_total': all_viol, 'c2_parity_max': out['c2_parity_max_over_triples'], 'written': str(outp)}, indent=2))
    print('\ntriple  base->ver   set   C0    transf decis score whitebx  Nspsa  viol')
    for r in complete:
        ct, h, s, c = (r['c0_transfer'], r['hsj'], r['spsa'], r['c2'])
        print(f"{r['triple']:5s} {r['base'][:4]:>4}->{r['verifier'][:4]:<4} {r['dataset']:4s} {ct['C0']['auroc']:.3f} {ct['transfer']['auroc']:.3f} {h['mean']:.3f} {s['mean']:.3f} {c['mean']:.3f}   {r['spsa_n']:4d}  {len(r['monotonicity_violations'])}")
if __name__ == '__main__':
    main()
