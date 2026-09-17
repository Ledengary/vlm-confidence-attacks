from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path
from src.config import DATA_DIR
from src.verifier_ladder.span12 import MSHORT, SPAN12, SSHORT, stim_dir
OUT = DATA_DIR / 'snowglobe' / 'verifier_escape'
import sys
PY = sys.executable
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
HSJ_Q = 1000
SPSA_Q = 600
SEEDS = [1, 2]
N = 200
VCOST = {'internvl3_5_2b': 3.0, 'gemma3_12b': 1.0, 'llava_onevision_7b': 0.9, 'molmo2_4b': 0.9}

def _ids(base, s):
    return OUT / f'sp_pool_{MSHORT[base]}_{SSHORT[s]}.json'
IV_VER = 'internvl3_5_2b'

def _is_iv_spsa(j):
    return j['kind'] == 'spsa' and j.get('verifier') == IV_VER

def _ids100(base, s):
    return OUT / f'sp_pool100_{MSHORT[base]}_{SSHORT[s]}.json'

def _spsa_cap_done(j):
    capf = _ids100(_base(j['key']), _set(j['key']))
    capids = set(json.load(open(capf))['item_ids'])
    have = set()
    for f in glob.glob(str(OUT / f"{j['key']}_query_seed{j['seed']}_s*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if 'ERROR' not in r and r.get('item_id') in capids:
                    have.add(r['item_id'])
    return len(have) >= len(capids)

def stim_done(base, s, full=False):
    d = OUT / 'matrix_full_images' / f'{MSHORT[base]}_{SSHORT[s]}' if full else Path(stim_dir(base, s))
    idsf = OUT / (f'sp_full_{MSHORT[base]}_{SSHORT[s]}.json' if full else f'sp_pool_{MSHORT[base]}_{SSHORT[s]}.json')
    ids = json.load(open(idsf))['item_ids']
    if not d.exists():
        return False
    ok = sum(((d / f'{i}__min__adv.png').exists() and (d / f'{i}__max__adv.png').exists() for i in ids))
    return ok >= (len(ids) if not full else int(0.98 * len(ids)))

def n_records(pat, need):
    ids = set()
    for f in glob.glob(str(OUT / pat)):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if 'ERROR' not in r and 'item_id' in r:
                    ids.add(r['item_id'])
    return len(ids) >= need

def c2_done(key, seed, need=N):
    ids = set()
    for f in glob.glob(str(OUT / f'c2_{key}_seed{seed}_scores_s*.jsonl')):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if 'ERROR' not in r and 'direction' in r:
                    ids.add((r['item_id'], r['direction']))
    return len(ids) >= 2 * need

def build_jobs():
    trips = {sp: (b, v, s) for sp, lab, b, v, s in SPAN12}
    jobs = []
    seen = set()
    for sp, (b, v, s) in trips.items():
        vc = VCOST[v]
        bs = (b, s)
        if bs not in seen:
            seen.add(bs)
            jobs.append({'kind': 'stim', 'base': b, 'set': s, 'id': f'stim_{MSHORT[b]}_{SSHORT[s]}', 'cost': N * 2})
            jobs.append({'kind': 'stim_full', 'base': b, 'set': s, 'id': f'stimF_{MSHORT[b]}_{SSHORT[s]}', 'cost': 1000 * 2})
        jobs.append({'kind': 'c0c1', 'key': sp, 'base': b, 'verifier': v, 'set': s, 'id': f'c0c1_{sp}', 'cost': N * vc})
        jobs.append({'kind': 'c0c1_full', 'key': sp, 'base': b, 'verifier': v, 'set': s, 'id': f'c0c1F_{sp}', 'cost': 1000 * vc})
        for sd in SEEDS:
            jobs.append({'kind': 'hsj', 'key': sp, 'seed': sd, 'verifier': v, 'id': f'hsj_{sp}_s{sd}', 'cost': N * HSJ_Q * vc * 0.3})
            spsa_n = 100 if v == 'internvl3_5_2b' else N
            jobs.append({'kind': 'spsa', 'key': sp, 'seed': sd, 'verifier': v, 'n': spsa_n, 'id': f'spsa_{sp}_s{sd}', 'cost': spsa_n * SPSA_Q * vc})
            jobs.append({'kind': 'c2', 'key': sp, 'seed': sd, 'base': b, 'verifier': v, 'set': s, 'id': f'c2_{sp}_s{sd}', 'cost': N * 2 * 31 * vc})
    return (jobs, trips)
TIER = {'stim': 0, 'c0c1': 1, 'hsj': 2, 'c2': 3, 'spsa': 4, 'stim_full': 5, 'c0c1_full': 6}

def is_done(j):
    k = j['kind']
    if k == 'stim':
        return stim_done(j['base'], j['set'])
    if k == 'stim_full':
        return stim_done(j['base'], j['set'], full=True)
    if k == 'c0c1':
        return n_records(f"{j['key']}_spscores_s*.jsonl", N)
    if k == 'c0c1_full':
        return n_records(f"{j['key']}_spscoresF_s*.jsonl", int(0.98 * 1000))
    if k == 'hsj':
        return n_records(f"{j['key']}_hardlabel_seed{j['seed']}_s*.jsonl", N)
    if k == 'spsa':
        if _is_iv_spsa(j):
            return _spsa_cap_done(j)
        return n_records(f"{j['key']}_query_seed{j['seed']}_s*.jsonl", N)
    if k == 'c2':
        return c2_done(j['key'], j['seed'])
    return False

def _progress(j):
    k = j['kind']
    try:
        if k in ('stim', 'stim_full'):
            full = k == 'stim_full'
            d = OUT / 'matrix_full_images' / f"{MSHORT[j['base']]}_{SSHORT[j['set']]}" if full else Path(stim_dir(j['base'], j['set']))
            return len(glob.glob(str(d / '*__adv.png')))
        if k == 'c2':
            n = 0
            for f in glob.glob(str(OUT / f"c2_{j['key']}_seed{j['seed']}_scores_s*.jsonl")):
                for l in open(f):
                    if l.strip() and 'ERROR' not in l and ('direction' in l):
                        n += 1
            return n
        pats = {'c0c1': f"{j.get('key')}_spscores_s*.jsonl", 'c0c1_full': f"{j.get('key')}_spscoresF_s*.jsonl", 'hsj': f"{j.get('key')}_hardlabel_seed{j.get('seed')}_s*.jsonl", 'spsa': f"{j.get('key')}_query_seed{j.get('seed')}_s*.jsonl"}
        n = 0
        for f in glob.glob(str(OUT / pats[k])):
            for l in open(f):
                if l.strip() and 'ERROR' not in l:
                    n += 1
        return n
    except Exception:
        return 0

def ready(j):
    if j['kind'] in ('c0c1',):
        return stim_done(j['base'], j['set'])
    if j['kind'] in ('c0c1_full',):
        return stim_done(j['base'], j['set'], full=True)
    return True

def cmd(j, gpu):
    k = j['kind']
    if k in ('stim', 'stim_full'):
        b, s = (j['base'], j['set'])
        full = k == 'stim_full'
        idsf = OUT / (f'sp_full_{MSHORT[b]}_{SSHORT[s]}.json' if full else f'sp_pool_{MSHORT[b]}_{SSHORT[s]}.json')
        imgd = OUT / 'matrix_full_images' / f'{MSHORT[b]}_{SSHORT[s]}' if full else Path(stim_dir(b, s))
        outd = OUT / ('matrix_full_base_attack' if full else 'matrix_base_attack') / f'{MSHORT[b]}_{SSHORT[s]}'
        return [PY, '-m', 'src.verifier_ladder.regenerate', '--model', b, '--set', s, '--gpu', str(gpu), '--items', str(idsf), '--out', str(outd), '--img-dir', str(imgd)]
    if k in ('c0c1', 'c0c1_full'):
        return [PY, '-m', 'src.verifier_ladder.span12_score', '--triple', j['key'], '--gpu', str(gpu), '--full' if k == 'c0c1_full' else '--common']
    if k == 'hsj':
        return [PY, '-m', 'src.verifier_ladder.hardlabel_attack', '--triple', j['key'], '--gpu', str(gpu), '--items', str(_ids(_base(j['key']), _set(j['key']))), '--max-q', str(HSJ_Q), '--seed', str(j['seed'])]
    if k == 'spsa':
        items = _ids100(_base(j['key']), _set(j['key'])) if _is_iv_spsa(j) else _ids(_base(j['key']), _set(j['key']))
        return [PY, '-m', 'src.verifier_ladder.query_attack', '--triple', j['key'], '--gpu', str(gpu), '--items', str(items), '--max-q', str(SPSA_Q), '--seed', str(j['seed'])]
    if k == 'c2':
        return [PY, '-m', 'src.verifier_ladder.c2_fill', '--gpu', str(gpu), '--items', str(_ids(j['base'], j['set'])), '--base', j['base'], '--verifier', j['verifier'], '--set', j['set'], '--suffix', f"_{j['key']}_seed{j['seed']}", '--seed', str(j['seed'])]
    raise ValueError(k)

def _base(sp):
    return {s[0]: s[2] for s in SPAN12}[sp]

def _set(sp):
    return {s[0]: s[4] for s in SPAN12}[sp]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-hours', type=float, default=72.0)
    a = ap.parse_args()
    logd = OUT / 'span12_logs'
    logd.mkdir(parents=True, exist_ok=True)
    start = float(os.environ.get('SP_START', '0')) or time.time()
    deadline = start + a.budget_hours * 3600
    jobs, trips = build_jobs()
    CHERRY = {'stim_full', 'c0c1_full'}

    def _sortkey(j):
        if j['kind'] in CHERRY:
            return (2, TIER[j['kind']], j['cost'])
        if j['kind'] == 'c2':
            return (0, j['cost'])
        if j['kind'] == 'spsa':
            return (1, -j['cost'])
        return (0, -j['cost'])
    jobs.sort(key=_sortkey)
    running, free = ({}, list(GPUS))
    attempts, quar, lastprog = ({}, set(), {})
    print(f'[span12] {len(jobs)} jobs on {len(GPUS)} GPUs, budget {a.budget_hours}h', flush=True)
    while True:
        for g in list(running):
            p, j = running[g]
            if p.poll() is not None:
                if not is_done(j):
                    pr = _progress(j)
                    if pr > lastprog.get(j['id'], -1):
                        lastprog[j['id']] = pr
                        attempts[j['id']] = 0
                    else:
                        attempts[j['id']] = attempts.get(j['id'], 0) + 1
                        if attempts[j['id']] >= 3:
                            quar.add(j['id'])
                            print(f"[span12] QUARANTINE {j['id']} (no progress)", flush=True)
                print(f"[span12] END {j['id']} rc={p.returncode} done={is_done(j)}", flush=True)
                free.append(g)
                del running[g]
        pend = [j for j in jobs if not is_done(j) and j['id'] not in quar]
        _write_progress(jobs, trips, running)
        if not pend and (not running):
            print('[span12] ALL DONE', flush=True)
            break
        if time.time() > deadline and (not running):
            print(f'[span12] budget exhausted; {len(pend)} jobs left', flush=True)
            break
        if time.time() <= deadline:
            busy = {j['id'] for _, j in running.values()}
            disp = [j for j in pend if ready(j) and j['id'] not in busy]
            while free and disp:
                j = disp.pop(0)
                g = free.pop(0)
                lf = open(logd / f"{j['id']}.log", 'a')
                env = dict(os.environ)
                for _tv in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[_tv] = '16'
                if j['kind'] == 'c2':
                    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                running[g] = (subprocess.Popen(cmd(j, g), stdout=lf, stderr=subprocess.STDOUT, env=env), j)
                print(f"[span12] START {j['id']} gpu{g}", flush=True)
        time.sleep(20)
    _write_progress(jobs, trips, running)

def _write_progress(jobs, trips, running):
    prog = {'running': {str(g): j['id'] for g, (p, j) in running.items()}, 'rungs': {}}
    for k in ('stim', 'c0c1', 'stim_full', 'c0c1_full', 'hsj', 'spsa', 'c2'):
        js = [j for j in jobs if j['kind'] == k]
        prog['rungs'][k] = {'done': sum((is_done(j) for j in js)), 'total': len(js)}
    comp = []
    for sp, lab, b, v, s in SPAN12:
        need = [j for j in jobs if j.get('key') == sp and j['kind'] in ('c0c1', 'hsj', 'spsa', 'c2')]
        comp.append({'triple': sp, 'label': lab, 'complete': all((is_done(j) for j in need))})
    prog['triples'] = comp
    prog['n_complete_to_standard'] = sum((c['complete'] for c in comp))
    json.dump(prog, open(OUT / 'span12_progress.json', 'w'), indent=2, default=str)
if __name__ == '__main__':
    main()
