from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.config import DATA_DIR, SEED
OUT = DATA_DIR / 'core_multiseed'
READOUTS = ('tokprob', 'pik', 'saplma', 'ccpsd', 'ccps', 'full', 'coupled')
OBJ_CONFIG = (('pik', 'B_corrected_30x2'), ('saplma', 'B_corrected_30x2'), ('ccpsd', 'B_corrected_30x2'), ('full', 'B_corrected_30x2'), ('coupled', 'C_corrected_60x3'))
DIRECTIONS = ('min', 'max')

def balanced_items(model, setname, n, seed):
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

def _install_seed(seed):
    from src.attacks import gate_v12
    orig = gate_v12.restart_seed
    if seed == 0:
        return
    mix = int(seed) * 2654435761 & 2147483647

    def seeded(item_id, objective, direction, r, _orig=orig, _mix=mix):
        return (_orig(item_id, objective, direction, r) ^ _mix) & 2147483647
    gate_v12.restart_seed = seeded

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--set', dest='setname', required=True)
    ap.add_argument('--celltag', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--deadline', type=float, default=0.0)
    a = ap.parse_args()
    from src.data_prep.canonical import load_manifest
    from src.inference.engine import build_item, load_adapter
    from src.attacks.official_pv import official_pv_fn_factory
    from src.estimators.objectives import forward_all
    from src.attacks import gate_v12
    from src.attacks.strict_bank import StrictCanonicalBank
    _install_seed(a.seed)
    dev = f'cuda:{a.gpu}'
    cell_dir = OUT / f'seed{a.seed}' / a.celltag
    cell_dir.mkdir(parents=True, exist_ok=True)
    rec_path = cell_dir / 'records.jsonl'
    done_ids = set()
    if rec_path.exists():
        for l in open(rec_path):
            if l.strip():
                try:
                    r = json.loads(l)
                    if 'ERROR' not in r:
                        done_ids.add(r['item_id'])
                except Exception:
                    pass
    items, k = balanced_items(a.model, a.setname, a.n, a.seed)
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
            print(f'[{a.celltag} s{a.seed}] deadline; stop at {it}', flush=True)
            break
        try:
            ctx = build_item(adapter, pp, manifest[row['item_id']], dev)
            off = official_pv_fn_factory(adapter, manifest[row['item_id']])
            out0 = forward_all(adapter, ctx, ctx.pv_official, need_grad=False)
            with torch.no_grad():
                clean_scores = {kk: None if v is None else float(v) for kk, v in bank.score_all(ctx, out0, row['item_id'], want=READOUTS).items()}
            rec = {'item_id': row['item_id'], 'label': int(row['label']), 'clean_scores': clean_scores, 'endpoints': {}}
            for obj, cfg in OBJ_CONFIG:
                for direction in DIRECTIONS:
                    r = gate_v12.run_one(adapter, bank, pp, ctx, row['item_id'], a.model, obj, direction, cfg, off)
                    rec['endpoints'][f'{obj}|{direction}'] = {'endpoint_scores': r['endpoint_scores'], 'used_clean_fallback': bool(r['used_clean_fallback']), 'linf_greylevels': r.get('validity', {}).get('linf_greylevels')}
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
            print(f'[{a.celltag} s{a.seed}] {n_ok + n_err} done ({n_err} err) {el / max(n_ok + n_err, 1):.1f}s/item', flush=True)
    fh.close()
    print(f'[{a.celltag} s{a.seed}] DONE ok={n_ok} err={n_err} {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    main()
