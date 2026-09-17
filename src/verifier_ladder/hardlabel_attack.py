from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.config import DATA_DIR, SEED
from src.attacks.attack import project, quantized_uint8
OUT = DATA_DIR / 'snowglobe' / 'verifier_escape'
EPS = 8.0 / 255.0
THRESH = 0.5
CHECKPOINTS = (200, 600, 1000, 5000)
PRES_EVERY = 25
INIT_TRIES = 150
GRAD_SAMPLES = 40

def _u8(x):
    return quantized_uint8(x).cpu().numpy().astype(np.uint8)

def run(triple_key: str, gpu: int, max_q: int, item_ids, shard: int=0, num_shards: int=1, checkpoints=CHECKPOINTS, preserve: bool=True, seed: int=0) -> dict:
    from src.data_prep.canonical import load_manifest
    from src.inference.engine import build_item, fresh_decode, load_adapter
    from src.attacks.official_pv import official_pv_fn_factory
    from src.verifier_ladder.data import load_cell
    from .triples import TRIPLES
    from .verifier import Verifier
    tri = TRIPLES[triple_key]
    dev = f'cuda:{gpu}'
    base_ad, base_pp, _ = load_adapter(tri['base'], gpu)
    ver_ad, _, _ = load_adapter(tri['verifier'], gpu)
    V = Verifier(ver_ad)
    manifest = {r['item_id']: r for r in load_manifest(tri['base'])}
    cell = load_cell(tri['base'], tri['set'])
    lab = {i: int(y) for i, y in zip(cell['ids'], cell['y'])}
    OUT.mkdir(parents=True, exist_ok=True)
    mine = item_ids[shard::num_shards]
    tagp = '' if preserve else '_nopres'
    stag = f'_seed{seed}' if seed else ''
    out_path = OUT / f'{triple_key}_hardlabel{tagp}{stag}_s{shard}.jsonl'
    import glob as _glob
    done = set()
    for _f in _glob.glob(str(OUT / f'{triple_key}_hardlabel{tagp}{stag}_s*.jsonl')):
        for line in open(_f):
            if line.strip():
                try:
                    _r = json.loads(line)
                    if 'item_id' in _r:
                        done.add(_r['item_id'])
                except Exception:
                    pass
    todo = [i for i in mine if i not in done]
    print(f'[{triple_key} hardlabel s{shard}] todo={len(todo)} max_q={max_q} thr={THRESH}', flush=True)
    fh = open(out_path, 'a')
    for iid in todo:
        row = manifest[iid]
        q, ans = (row['question'], row['committed_clean_answer'])
        y = lab[iid]
        target_accept = y == 0
        try:
            ctx = build_item(base_ad, base_pp, row, dev)
            off = official_pv_fn_factory(base_ad, row)
            committed = ctx.row['committed_clean_answer']
            x0 = ctx.raw01.detach().to(torch.float32)
            clean_u8 = _u8(x0)
            gen = torch.Generator(device=dev)
            gen.manual_seed(SEED + seed * 1000003 + hash(iid) % 100000)
            queries = [0]

            def score_u8(u8):
                img = Image.fromarray(np.transpose(u8, (1, 2, 0)), 'RGB')
                return V.yes_prob(img, q, ans)['score']

            def is_target_x(x):
                queries[0] += 1
                acc = score_u8(_u8(x)) >= THRESH
                return acc == target_accept

            def preserved(u8):
                text, _ = fresh_decode(base_ad, ctx, off(torch.from_numpy(u8)).to(dev))
                return text == committed
            clean_score = score_u8(clean_u8)
            clean_is_target = (clean_score >= THRESH) == target_accept
            best_u8 = clean_u8
            q_first_flip = None
            ck = sorted((c for c in checkpoints if c <= max_q))
            ckpt_scores = {}

            def record_ckpts():
                while ck and queries[0] >= ck[0]:
                    ckpt_scores[str(ck[0])] = score_u8(best_u8)
                    ck.pop(0)
            if clean_is_target:
                q_first_flip = 0
                best_u8 = clean_u8
            else:
                x_adv = None
                for _ in range(INIT_TRIES):
                    if queries[0] >= max_q:
                        break
                    d = (torch.randint(0, 2, x0.shape, generator=gen, device=dev, dtype=torch.float32) * 2 - 1) * EPS
                    xc = project(x0 + d, x0, EPS)
                    if is_target_x(xc):
                        x_adv = xc.detach()
                        break
                if x_adv is not None:
                    x_cur = x_adv
                    it = 0
                    while queries[0] < max_q:
                        it += 1
                        lo, hi = (x0.clone(), x_cur.clone())
                        for _ in range(8):
                            if queries[0] >= max_q:
                                break
                            mid = (lo + hi) / 2
                            if is_target_x(mid):
                                hi = mid
                            else:
                                lo = mid
                        x_b = hi
                        delta = 0.01 * EPS * (it ** (-0.5) + 1.0)
                        acc = torch.zeros_like(x0)
                        nb = 0
                        for _ in range(GRAD_SAMPLES):
                            if queries[0] >= max_q:
                                break
                            u = torch.randint(0, 2, x0.shape, generator=gen, device=dev, dtype=torch.float32) * 2 - 1
                            phi = 1.0 if is_target_x(project(x_b + delta * u, x0, EPS)) else -1.0
                            acc = acc + phi * u
                            nb += 1
                        if nb == 0:
                            break
                        gdir = torch.sign(acc / nb)
                        step = EPS
                        moved = False
                        for _ in range(6):
                            if queries[0] >= max_q:
                                break
                            xt = project(x_b + step * gdir, x0, EPS)
                            if is_target_x(xt):
                                x_cur = xt.detach()
                                moved = True
                                break
                            step /= 2.0
                        if not moved:
                            x_cur = x_b.detach()
                        cand = _u8(x_cur)
                        if not preserve:
                            best_u8 = cand
                            if q_first_flip is None:
                                q_first_flip = queries[0]
                        elif it % PRES_EVERY == 0 or q_first_flip is None:
                            if preserved(cand):
                                best_u8 = cand
                                if q_first_flip is None:
                                    q_first_flip = queries[0]
                        record_ckpts()
            record_ckpts()
            for c in sorted((cc for cc in checkpoints if cc <= max_q)):
                ckpt_scores.setdefault(str(c), score_u8(best_u8))
            gl = int(np.abs(best_u8.astype(np.int32) - clean_u8.astype(np.int32)).max())
            final_target = (score_u8(best_u8) >= THRESH) == target_accept
            fh.write(json.dumps({'item_id': iid, 'label': y, 'threshold': THRESH, 'target_accept': bool(target_accept), 'clean_score': clean_score, 'clean_is_target': bool(clean_is_target), 'checkpoint_scores': ckpt_scores, 'flipped': bool(final_target), 'q_first_flip': q_first_flip, 'final_linf_greylevels': gl, 'used_clean': bool(np.array_equal(best_u8, clean_u8)), 'total_queries': queries[0]}) + '\n')
            fh.flush()
            print(f'  [{triple_key} hl s{shard}] {iid} flip={final_target} qff={q_first_flip} ckpt={ckpt_scores}', flush=True)
        except Exception as e:
            import traceback
            fh.write(json.dumps({'item_id': iid, 'ERROR': str(e)[:200], 'trace': traceback.format_exc()[-300:]}) + '\n')
            fh.flush()
    fh.close()
    return {'done': len(todo)}

def analyze(triple_key: str, tagp: str='') -> dict:
    import glob
    from src.verifier_ladder.data import auroc
    rows = []
    for f in sorted(glob.glob(str(OUT / f'{triple_key}_hardlabel{tagp}_s*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if 'ERROR' not in r:
                    rows.append(r)
    by = {r['item_id']: r for r in rows}
    y = np.array([r['label'] for r in by.values()])
    cks = sorted({int(c) for r in by.values() for c in r['checkpoint_scores']})
    out = {'triple': triple_key, 'n': len(y), 'threshold': THRESH, 'budgets': {}}
    for c in cks:
        s = np.array([r['checkpoint_scores'].get(str(c), list(r['checkpoint_scores'].values())[-1]) for r in by.values()])
        out['budgets'][str(c)] = round(auroc(y, s), 4)
    flips = [r['flipped'] for r in by.values()]
    qff = [r['q_first_flip'] for r in by.values() if r['q_first_flip'] is not None]
    out['flip_rate'] = round(float(np.mean(flips)), 4)
    out['n_flipped'] = int(np.sum(flips))
    out['queries_to_first_flip'] = {'n_with_flip': len(qff), 'median': int(np.median(qff)) if qff else None, 'min': int(np.min(qff)) if qff else None, 'max': int(np.max(qff)) if qff else None, 'n_clean_already_target': int(sum((1 for r in by.values() if r['clean_is_target'])))}
    out['mean_final_greylevels'] = round(float(np.mean([r['final_linf_greylevels'] for r in by.values()])), 2)
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--triple', default='t0_llava_gemma_vqa')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max-q', type=int, default=1000)
    ap.add_argument('--items', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--no-preserve', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.analyze:
        print(json.dumps(analyze(a.triple, '' if not a.no_preserve else '_nopres'), indent=2))
    else:
        ids = json.load(open(a.items))['item_ids']
        print(json.dumps(run(a.triple, a.gpu, a.max_q, ids, a.shard, a.num_shards, preserve=not a.no_preserve, seed=a.seed), indent=2))
