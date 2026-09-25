from __future__ import annotations
import ast
import glob
import json
from pathlib import Path
import numpy as np
from src.config import DATA_DIR
COUPLED = DATA_DIR / 'coupled'
MANIFEST = DATA_DIR / 'iclr2027' / 'manifest'
DRAWS = DATA_DIR / 'draws'
OUT = DATA_DIR / 'snowglobe' / 'frontier_lambda'
FEATURE_KEY = 'h_sap'
FEATURE_KEY_ADV = 'h_sap_adv'
LOCUS = 'saplma locus, last hidden state of the final answer token, float32, no f16 round trip'

def _source_image_id(setname: str, draw: dict) -> str:
    if setname == 'gqa_val_eval':
        return str(draw['image_key'])
    meta = draw.get('meta')
    if isinstance(meta, str):
        meta = ast.literal_eval(meta)
    return str((meta or {}).get('coco_image_id', draw['image_key']))

def load_cell(model: str, setname: str) -> dict:
    feats, feats_adv, ids = ([], [], [])
    for f in sorted(glob.glob(str(COUPLED / model / f'{setname}_hidden_s*.npz'))):
        z = np.load(f, allow_pickle=True)
        if FEATURE_KEY_ADV not in z.files:
            raise RuntimeError(f'{f} has no {FEATURE_KEY_ADV}; this cell has no stored endpoint')
        ids.append(np.asarray([str(x) for x in z['ids']]))
        feats.append(z[FEATURE_KEY].astype(np.float32))
        feats_adv.append(z[FEATURE_KEY_ADV].astype(np.float32))
    ids = np.concatenate(ids)
    X = np.concatenate(feats)
    Xa = np.concatenate(feats_adv)
    meta = {}
    for f in sorted(glob.glob(str(COUPLED / model / f'{setname}_meta_s*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                meta[r['item_id']] = r
    man = {}
    for line in open(MANIFEST / f'{model}.jsonl'):
        r = json.loads(line)
        if r['dataset'] == setname:
            man[r['item_id']] = r
    draws = {}
    for line in open(DRAWS / f'{setname}.jsonl'):
        if line.strip():
            r = json.loads(line)
            draws[r['item_id']] = r
    keep = [i for i, iid in enumerate(ids) if iid in meta and iid in man and (iid in draws)]
    ids = ids[keep]
    X, Xa = (X[keep], Xa[keep])
    y = np.array([int(meta[i]['label']) for i in ids], dtype=np.int64)
    answers = np.array([str(man[i]['committed_clean_answer']) for i in ids])
    img = np.array([_source_image_id(setname, draws[i]) for i in ids])
    found = np.array([bool(meta[i].get('found')) for i in ids])
    byte_ok = np.array([bool(meta[i].get('byte_identical')) for i in ids])
    moved = np.abs(X - Xa).max(axis=1) > 0
    return {'model': model, 'dataset': setname, 'n': len(ids), 'ids': ids, 'X': X, 'X_adv': Xa, 'y': y, 'answers': answers, 'image_id': img, 'found': found, 'byte_identical': byte_ok, 'feature_moved': moved, 'dim': int(X.shape[1]), 'n_correct': int((y == 1).sum()), 'n_wrong': int((y == 0).sum()), 'engaged_rate': float(found.mean()), 'feature_moved_rate': float(moved.mean()), 'n_source_images': int(len(set(img))), 'clean_png': {i: man[i]['clean_png'] for i in ids}, 'locus': LOCUS, 'endpoint_provenance': 'one stored answer-preserving endpoint per item from the coupled extraction stage: raw-pixel L-inf 8/255, 15 steps, alpha eps/4, tau_safe 1.0, label-directed correct-down / wrong-up, best answer-preserving step, byte identity verified by fresh greedy re-decode'}

def image_disjoint_split(cell: dict, seed: int=23, eval_frac: float=0.4) -> dict:
    rng = np.random.default_rng(seed)
    groups = sorted(set(cell['image_id']))
    rng.shuffle(groups)
    target = int(round(eval_frac * cell['n']))
    ev_groups, n = (set(), 0)
    for g in groups:
        if n >= target:
            break
        ev_groups.add(g)
        n += int((cell['image_id'] == g).sum())
    ev = np.array([g in ev_groups for g in cell['image_id']])
    tr = ~ev
    overlap = set(cell['image_id'][tr]) & set(cell['image_id'][ev])
    assert not overlap, f'image leakage: {len(overlap)} shared source images'
    return {'train_idx': np.where(tr)[0], 'eval_idx': np.where(ev)[0], 'n_train': int(tr.sum()), 'n_eval': int(ev.sum()), 'train_correct': int((cell['y'][tr] == 1).sum()), 'train_wrong': int((cell['y'][tr] == 0).sum()), 'eval_correct': int((cell['y'][ev] == 1).sum()), 'eval_wrong': int((cell['y'][ev] == 0).sum()), 'train_images': len(set(cell['image_id'][tr])), 'eval_images': len(set(cell['image_id'][ev])), 'shared_images': 0, 'seed': seed, 'eval_frac': eval_frac, 'rule': 'disjoint on source image id (coco_image_id for VQAv2/POPE, image_key for GQA)'}

def answer_string_prior(cell: dict, idx=None) -> float:
    from collections import defaultdict
    i = np.arange(cell['n']) if idx is None else idx
    a, y = (cell['answers'][i], cell['y'][i])
    tot, pos = (defaultdict(int), defaultdict(int))
    for s, l in zip(a, y):
        tot[s] += 1
        pos[s] += int(l)
    score = np.array([pos[s] / tot[s] for s in a])
    return auroc(y, score)

def auroc(y, s) -> float:
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    y, s = (y[m], s[m])
    if len(set(y.tolist())) < 2:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), float)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1, n0 = (float((y == 1).sum()), float((y == 0).sum()))
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='llava_onevision_7b')
    ap.add_argument('--set', dest='setname', default='vqav2_ood')
    a = ap.parse_args()
    c = load_cell(a.model, a.setname)
    sp = image_disjoint_split(c)
    print(json.dumps({k: v for k, v in c.items() if k in ('model', 'dataset', 'n', 'dim', 'n_correct', 'n_wrong', 'engaged_rate', 'feature_moved_rate', 'n_source_images')}, indent=2))
    print(json.dumps({k: v for k, v in sp.items() if not k.endswith('_idx')}, indent=2))
    print('answer-string prior c* (all):', round(answer_string_prior(c), 4))
    print('answer-string prior c* (eval split):', round(answer_string_prior(c, sp['eval_idx']), 4))
