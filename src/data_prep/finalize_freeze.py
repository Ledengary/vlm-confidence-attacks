from __future__ import annotations
import glob
import json
from collections import defaultdict
import numpy as np
from src.config import MANIFESTS_DIR, TABLES_DIR
from src.data_prep.final_freeze import FREEZE_SETS, RECORDS_DIR, DRAW_DIR
FINAL_DIR = MANIFESTS_DIR / 'final'
GQA_TRAIN_TRAIN, GQA_TRAIN_VAL = (20000, 5000)

def _load_records(name):
    recs = []
    for f in sorted(glob.glob(str(RECORDS_DIR / name / 'shard_*.jsonl'))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                if 'error' not in r:
                    recs.append(r)
    recs.sort(key=lambda r: r['draw_index'])
    return recs

def _gpt_calls(name):
    tot = 0
    for f in glob.glob(str(RECORDS_DIR / name / 'shard_*.gpt.json')):
        tot += json.loads(open(f).read()).get('n_calls', 0)
    return tot

def _draw_strata(name):
    frac = defaultdict(int)
    n = 0
    for line in open(DRAW_DIR / f'{name}.jsonl'):
        if line.strip():
            frac[json.loads(line)['answer_type']] += 1
            n += 1
    return ({k: v / n for k, v in frac.items()}, n)

def _stats(frozen, draw_strata):
    ious = [r['relevant']['iou_native_matched'] for r in frozen if r.get('relevant') and r['relevant'].get('iou_native_matched') is not None]
    rel_area = [r['relevant'].get('area_frac') for r in frozen if r.get('relevant') and r['relevant'].get('area_frac') is not None]
    comp_area = [r['competing'].get('area_frac') for r in frozen if r.get('competing') and r['competing'].get('area_frac') is not None]
    vmix = defaultdict(int)
    for r in frozen:
        c = r.get('competing')
        if c:
            vmix[c.get('variant', '?')] += 1
    fz = defaultdict(int)
    for r in frozen:
        fz[r['answer_type']] += 1
    nf = max(1, len(frozen))
    max_dev = 0.0
    for k in set(list(draw_strata) + list(fz)):
        max_dev = max(max_dev, abs(draw_strata.get(k, 0) - fz.get(k, 0) / nf))

    def d(a):
        a = np.array(a, dtype=float)
        return {} if a.size == 0 else {'n': int(a.size), 'mean': round(float(a.mean()), 4), 'median': round(float(np.median(a)), 4)}
    return {'relevant_iou': d(ious), 'relevant_area': d(rel_area), 'competing_area': d(comp_area), 'variant_mix': dict(vmix), 'variant_B_frac': round(vmix.get('B', 0) / max(1, sum(vmix.values())), 4), 'stratification_max_dev': round(max_dev, 4), 'strata_frozen': {k: round(v / nf, 4) for k, v in sorted(fz.items())}}

def _write_manifest(name, frozen):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_DIR / f'{name}.jsonl', 'w') as f:
        for r in frozen:
            f.write(json.dumps({'item_id': r['item_id'], 'answer_type': r['answer_type'], 'lang': r['lang'], 'dataset': r['dataset'], 'split': r['split'], 'relevant': r.get('relevant'), 'competing': r.get('competing')}) + '\n')

def finalize():
    out = TABLES_DIR / '004_final_freeze'
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    raw_dist = {}
    total_gpt = 0
    for name, dataset, split, target, mult, _req in FREEZE_SETS:
        recs = _load_records(name)
        gpt = _gpt_calls(name)
        total_gpt += gpt
        draw_strata, draw_n = _draw_strata(name)
        usable = [r for r in recs if r.get('usable')]
        n_gate = sum((1 for r in recs if r.get('gate_pass')))
        if target is None:
            frozen = usable
            raw = len(recs)
            shortfall = 0
            multiplier = round(raw / max(1, len(frozen)), 3)
            _write_manifest(name, frozen)
        else:
            if len(usable) >= target:
                frozen = usable[:target]
                raw = frozen[-1]['draw_index'] + 1
                shortfall = 0
            else:
                frozen = usable
                raw = len(recs)
                shortfall = target - len(usable)
            multiplier = round(raw / max(1, len(frozen)), 3)
            if name == 'gqa_train_probe':
                _write_manifest('gqa_train_probe_train', frozen[:GQA_TRAIN_TRAIN])
                _write_manifest('gqa_train_probe_val', frozen[GQA_TRAIN_TRAIN:GQA_TRAIN_TRAIN + GQA_TRAIN_VAL])
            else:
                _write_manifest(name, frozen)
        st = _stats(frozen, draw_strata)
        summary[name] = {'dataset': dataset, 'split': split, 'target': target, 'usable_frozen': len(frozen), 'raw_processed': raw, 'raw_drawn': draw_n, 'raw_to_usable_multiplier': multiplier, 'shortfall': shortfall, 'gate_pass_of_processed': round(n_gate / max(1, len(recs)), 4), 'verification_pass_of_gated': round(len(usable) / max(1, n_gate), 4), 'gpt_calls': gpt, **st}
        raw_dist[name] = {'relevant_iou': [r['relevant']['iou_native_matched'] for r in frozen if r.get('relevant') and r['relevant'].get('iou_native_matched') is not None], 'relevant_area': [r['relevant']['area_frac'] for r in frozen if r.get('relevant') and r['relevant'].get('area_frac') is not None], 'competing_area': [r['competing']['area_frac'] for r in frozen if r.get('competing') and r['competing'].get('area_frac') is not None]}
        print(f"{name}: frozen {len(frozen)}/{target} usable, raw {raw}, mult {multiplier}, relIoU~{st['relevant_iou'].get('mean')}, B-frac {st['variant_B_frac']}, strat_dev {st['stratification_max_dev']}, gpt {gpt}", flush=True)
    summary['_totals'] = {'gpt_calls': total_gpt, 'gpt_cost_estimate_usd_note': 'see report for cost model'}
    with open(out / 'final_freeze_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(out / 'raw_dist.json', 'w') as f:
        json.dump(raw_dist, f)
    print(f"\ntotal gpt calls: {total_gpt}. wrote {out / 'final_freeze_summary.json'}")
    return summary
if __name__ == '__main__':
    finalize()
