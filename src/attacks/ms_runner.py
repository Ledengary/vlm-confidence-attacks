from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path
from src.config import DATA_DIR
OUT = DATA_DIR / 'core_multiseed'
import sys
PY = sys.executable
GPUS = [0, 1, 2, 3, 4, 6, 7]
SEEDS = [24, 25]
N = 120
MODELS = ['internvl3_5_2b', 'gemma3_12b', 'molmo2_4b', 'llava_onevision_7b']
SETS = ['gqa_val_eval', 'vqav2_ood', 'pope_ood']
MSHORT = {'internvl3_5_2b': 'internvl', 'gemma3_12b': 'gemma', 'molmo2_4b': 'molmo', 'llava_onevision_7b': 'llava'}
SSHORT = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqa', 'pope_ood': 'pope'}
PERITEM = {'internvl3_5_2b': 401, 'gemma3_12b': 251, 'molmo2_4b': 140, 'llava_onevision_7b': 126}

def celltag(m, s):
    return f'{MSHORT[m]}_{SSHORT[s]}'

def build_jobs():
    jobs = []
    for sd in SEEDS:
        for m in MODELS:
            for s in SETS:
                jobs.append({'model': m, 'set': s, 'seed': sd, 'tag': celltag(m, s), 'id': f'{celltag(m, s)}_s{sd}', 'cost': PERITEM[m] * N})
    return jobs

def n_done(j):
    p = OUT / f"seed{j['seed']}" / j['tag'] / 'records.jsonl'
    if not p.exists():
        return 0
    ids = set()
    for l in open(p):
        if l.strip():
            try:
                r = json.loads(l)
            except Exception:
                continue
            if 'ERROR' not in r and 'item_id' in r:
                ids.add(r['item_id'])
    return len(ids)

def target_n(j):
    return N

def is_done(j):
    return n_done(j) >= target_n(j) - 1

def cmd(j, gpu):
    return [PY, '-m', 'src.attacks.run_cell_ms', '--model', j['model'], '--set', j['set'], '--celltag', j['tag'], '--gpu', str(gpu), '--n', str(N), '--seed', str(j['seed'])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-hours', type=float, default=48.0)
    a = ap.parse_args()
    logd = OUT / 'logs'
    logd.mkdir(parents=True, exist_ok=True)
    start = float(os.environ.get('MS_START', '0')) or time.time()
    deadline = start + a.budget_hours * 3600
    jobs = build_jobs()
    jobs.sort(key=lambda j: -j['cost'])
    running, free = ({}, list(GPUS))
    attempts, quar, lastprog = ({}, set(), {})
    print(f'[ms] {len(jobs)} jobs on {len(GPUS)} GPUs, budget {a.budget_hours}h', flush=True)
    while True:
        for g in list(running):
            p, j = running[g]
            if p.poll() is not None:
                if not is_done(j):
                    pr = n_done(j)
                    if pr > lastprog.get(j['id'], -1):
                        lastprog[j['id']] = pr
                        attempts[j['id']] = 0
                    else:
                        attempts[j['id']] = attempts.get(j['id'], 0) + 1
                        if attempts[j['id']] >= 3:
                            quar.add(j['id'])
                            print(f"[ms] QUARANTINE {j['id']} (no progress)", flush=True)
                print(f"[ms] END {j['id']} rc={p.returncode} done={is_done(j)} n={n_done(j)}", flush=True)
                free.append(g)
                del running[g]
        pend = [j for j in jobs if not is_done(j) and j['id'] not in quar]
        _progress(jobs, running)
        if not pend and (not running):
            print('[ms] ALL DONE', flush=True)
            break
        if time.time() > deadline and (not running):
            print(f'[ms] budget exhausted; {len(pend)} left', flush=True)
            break
        if time.time() <= deadline:
            busy = {j['id'] for _, j in running.values()}
            disp = [j for j in pend if j['id'] not in busy]
            while free and disp:
                j = disp.pop(0)
                g = free.pop(0)
                lf = open(logd / f"{j['id']}.log", 'a')
                env = dict(os.environ)
                for tv in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[tv] = '16'
                running[g] = (subprocess.Popen(cmd(j, g), stdout=lf, stderr=subprocess.STDOUT, env=env), j)
                print(f"[ms] START {j['id']} gpu{g}", flush=True)
        time.sleep(20)
    _progress(jobs, running)

def _progress(jobs, running):
    prog = {'running': {str(g): j['id'] for g, (p, j) in running.items()}, 'cells': {j['id']: {'n': n_done(j), 'target': target_n(j), 'done': is_done(j)} for j in jobs}, 'n_done': sum((is_done(j) for j in jobs)), 'n_total': len(jobs)}
    json.dump(prog, open(OUT / 'ms_progress.json', 'w'), indent=2)
if __name__ == '__main__':
    main()
