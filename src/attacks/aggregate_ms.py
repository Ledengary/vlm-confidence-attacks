from __future__ import annotations
import glob
import json
from collections import defaultdict
import numpy as np
from src.config import DATA_DIR
from src.attacks.definitions import auroc, sel_witnessed_union_oracle
OUT = DATA_DIR / 'core_multiseed'
FROZEN = DATA_DIR / 'snowglobe' / 'pfmc_v1_2_2' / 'full_matrix_runs'
READOUTS = ['tokprob', 'pik', 'saplma', 'ccpsd', 'ccps', 'full', 'coupled']
MODELS = ['internvl3_5_2b', 'gemma3_12b', 'molmo2_4b', 'llava_onevision_7b']
SETS = ['gqa_val_eval', 'vqav2_ood', 'pope_ood']
MSHORT = {'internvl3_5_2b': 'internvl', 'gemma3_12b': 'gemma', 'molmo2_4b': 'molmo', 'llava_onevision_7b': 'llava'}
SSHORT = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqa', 'pope_ood': 'pope'}

def _cand_matrix(recs, R):
    rows = []
    for r in recs:
        c = [r['clean_scores'].get(R)] + [ep['endpoint_scores'].get(R) for ep in r['endpoints'].values()]
        rows.append(c)
    return np.array(rows, float)

def cell_metrics(recs):
    y = np.array([r['label'] for r in recs])
    per = {}
    binding_scores = {}
    for R in READOUTS:
        cand = _cand_matrix(recs, R)
        s = sel_witnessed_union_oracle(cand, y)
        per[R] = {'attacked': round(auroc(s, y), 4), 'clean': round(auroc(cand[:, 0], y), 4)}
        binding_scores[R] = s
    binding = max(per, key=lambda R: per[R]['attacked'])
    return {'n': len(y), 'n_correct': int((y == 1).sum()), 'n_wrong': int((y == 0).sum()), 'binding_readout': binding, 'attacked_auroc': per[binding]['attacked'], 'clean_auroc_binding': per[binding]['clean'], 'clean_auroc_median7': round(float(np.median([per[R]['clean'] for R in READOUTS])), 4), 'per_readout': per, '_y': y, '_binding_scores': binding_scores[binding]}

def boot_ci(y, s, n=2000, seed=23):
    rng = np.random.default_rng(seed)
    m = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(auroc(s[idx], y[idx]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

def load_records(path):
    recs = []
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if 'ERROR' not in r and 'clean_scores' in r:
                recs.append(r)
    return recs

def frozen_seed23_cell(model, dataset):
    byitem = defaultdict(list)
    label = {}
    for f in glob.glob(str(FROZEN / f'{model}_s*.jsonl')):
        for l in open(f):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get('dataset') != dataset or 'endpoint_scores' not in r or 'clean_scores' not in r:
                continue
            byitem[r['item_id']].append(r)
            label[r['item_id']] = int(r['label'])
    recs = []
    for it, rs in byitem.items():
        recs.append({'label': label[it], 'clean_scores': rs[0]['clean_scores'], 'endpoints': {f"{x['objective']}|{x['direction']}": {'endpoint_scores': x['endpoint_scores']} for x in rs}})
    return recs

def main():
    results = []
    for model in MODELS:
        for dataset in SETS:
            tag = f'{MSHORT[model]}_{SSHORT[dataset]}'
            cell = {'cell': f'{model}/{dataset}', 'tag': tag, 'seeds': {}}
            f23 = frozen_seed23_cell(model, dataset)
            if f23:
                m = cell_metrics(f23)
                cell['seeds']['23'] = {'n': m['n'], 'attacked_auroc': m['attacked_auroc'], 'clean_auroc_binding': m['clean_auroc_binding'], 'binding_readout': m['binding_readout'], 'ci': boot_ci(m['_y'], m['_binding_scores']), 'source': 'frozen_n1000'}
            for sd in (24, 25):
                p = OUT / f'seed{sd}' / tag / 'records.jsonl'
                if not p.exists():
                    continue
                recs = load_records(p)
                if len(recs) < 10:
                    continue
                m = cell_metrics(recs)
                cell['seeds'][str(sd)] = {'n': m['n'], 'attacked_auroc': m['attacked_auroc'], 'clean_auroc_binding': m['clean_auroc_binding'], 'binding_readout': m['binding_readout'], 'ci': boot_ci(m['_y'], m['_binding_scores']), 'source': 'reseed_n120'}
            av = [cell['seeds'][s]['attacked_auroc'] for s in cell['seeds']]
            cv = [cell['seeds'][s]['clean_auroc_binding'] for s in cell['seeds']]
            if av:
                cell['attacked_mean'] = round(float(np.mean(av)), 4)
                cell['attacked_sd'] = round(float(np.std(av)), 4)
                cell['clean_mean'] = round(float(np.mean(cv)), 4)
                cell['n_seeds'] = len(av)
                cell['all_below_chance'] = bool(max(av) < 0.5)
            results.append(cell)
    complete = [c for c in results if c.get('n_seeds', 0) >= 2]
    survivor = next((c for c in results if c['tag'] == 'molmo_pope'), None)
    out = {'n_cells': len(results), 'n_with_multiseed': len(complete), 'max_attacked_sd': round(max([c.get('attacked_sd', 0) for c in complete], default=0), 4), 'cells_all_below_chance': sum((c.get('all_below_chance', False) for c in complete)), 'survivor_molmo_pope': survivor['seeds'] if survivor else None, 'cells': results}
    json.dump(out, open(OUT / 'ms_results.json', 'w'), indent=2, default=str)
    print(json.dumps({k: out[k] for k in ('n_cells', 'n_with_multiseed', 'max_attacked_sd', 'cells_all_below_chance')}, indent=2))
    print(f"\n{'cell':26s} {'bind':>7} {'s23':>6} {'s24':>6} {'s25':>6} {'mean':>6} {'sd':>6} {'clean':>6}")
    for c in results:
        s = c['seeds']

        def g(k):
            return f"{s[k]['attacked_auroc']:.3f}" if k in s else '  -  '
        print(f"{c['tag']:26s} {s.get('23', {}).get('binding_readout', '?')[:6]:>7} {g('23')} {g('24')} {g('25')} {c.get('attacked_mean', 0):.3f} {c.get('attacked_sd', 0):.3f} {c.get('clean_mean', 0):.3f}")
    if survivor:
        print(f'\nSURVIVOR molmo/pope (coupled): ' + ' '.join((f"s{k}={survivor['seeds'][k]['attacked_auroc']:.3f}" for k in survivor['seeds'])))
if __name__ == '__main__':
    main()
