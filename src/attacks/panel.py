from __future__ import annotations
import glob
import json
import random
import numpy as np
from src.config import DATA_DIR, SEED
from src.inference.families import MODELS, SETS
RECON = DATA_DIR / 'iclr2027' / 'gates' / 'clean_recon'
OUT = DATA_DIR / 'iclr2027' / 'panel'
N_PER_CELL = 8

def _recon_rows(model: str) -> dict:
    rows = {}
    for f in sorted(glob.glob(str(RECON / f'{model}_s*.jsonl'))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                if 'ERROR' not in r:
                    rows[r['item_id']] = r
    return rows

def _split_median(items, keyfn):
    vals = np.array([keyfn(r) for r in items], dtype=np.float64)
    med = float(np.median(vals))
    lo = [r for r in items if keyfn(r) <= med]
    hi = [r for r in items if keyfn(r) > med]
    return (lo, hi, med)

def select_cell(rows: list, rng: random.Random) -> tuple:
    picked = []
    prov = {'strata': []}
    for label, group in (('correct', [r for r in rows if r['label'] == 1]), ('incorrect', [r for r in rows if r['label'] == 0])):
        short, long_, len_med = _split_median(group, lambda r: r['n_span_tokens'])
        fallback = False
        if not short or not long_:
            fallback = True
            short, long_, len_med = _split_median(group, lambda r: r['clean_min_margin'])
        for lname, sub in (('short', short), ('long', long_)):
            lo, hi, mmed = _split_median(sub, lambda r: r['clean_min_margin'])
            for mname, stratum in (('low_margin', lo), ('high_margin', hi)):
                pool = sorted(stratum, key=lambda r: r['item_id'])
                if not pool:
                    pool = sorted(sub, key=lambda r: r['item_id'])
                choice = rng.choice(pool)
                picked.append(choice)
                prov['strata'].append({'correctness': label, 'length_stratum': lname, 'margin_stratum': mname, 'length_split_fallback_to_margin': fallback, 'length_median': len_med, 'margin_median': mmed, 'stratum_size': len(stratum), 'item_id': choice['item_id'], 'n_span_tokens': choice['n_span_tokens'], 'clean_min_margin': choice['clean_min_margin'], 'clean_tokprob': choice['tokprob_fresh']})
    return (picked, prov)

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = []
    provenance = {}
    for model in MODELS:
        recon = _recon_rows(model)
        for s in SETS:
            rows = [r for r in recon.values() if r['dataset'] == s]
            if len(rows) != 1000:
                print(f'WARNING {model}/{s}: {len(rows)} reconstructed items, expected 1000')
            rng = random.Random(SEED)
            picked, prov = select_cell(rows, rng)
            assert len(picked) == N_PER_CELL, (model, s, len(picked))
            ids = [p['item_id'] for p in picked]
            assert len(set(ids)) == N_PER_CELL, f'{model}/{s}: duplicate panel item'
            provenance[f'{model}|{s}'] = prov
            for p in picked:
                panel.append({'model': model, 'dataset': s, 'item_id': p['item_id'], 'label': p['label'], 'group': p['group'], 'n_span_tokens': p['n_span_tokens'], 'clean_min_margin': p['clean_min_margin'], 'clean_tokprob': p['tokprob_fresh'], 'clean_answer': p['committed_answer']})
    with open(OUT / 'panel_96.jsonl', 'w') as fh:
        for r in panel:
            fh.write(json.dumps(r) + '\n')
    json.dump({'rule': __doc__, 'seed': SEED, 'n_per_cell': N_PER_CELL, 'n_total': len(panel), 'provenance': provenance}, open(OUT / 'panel_96_provenance.json', 'w'), indent=2)
    print(f"panel: {len(panel)} items over {len(set(((r['model'], r['dataset']) for r in panel)))} cells")
    for model in MODELS:
        for s in SETS:
            rr = [r for r in panel if r['model'] == model and r['dataset'] == s]
            print(f"  {model:20s} {s:14s} n={len(rr)} correct={sum((1 for r in rr if r['label'] == 1))} span_tokens={sorted(set((r['n_span_tokens'] for r in rr)))} margin=[{min((r['clean_min_margin'] for r in rr)):.2f},{max((r['clean_min_margin'] for r in rr)):.2f}]")
    return panel

def load_panel() -> list:
    return [json.loads(l) for l in open(OUT / 'panel_96.jsonl') if l.strip()]
if __name__ == '__main__':
    build()
