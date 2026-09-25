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
SPSA_C = 2.0 / 255.0
SPSA_LR = 1.0 / 255.0
CHECKPOINTS = (200, 600, 1000, 5000)
PRES_EVERY = 20

def _u8(x):
    return quantized_uint8(x).cpu().numpy().astype(np.uint8)

def run(triple_key: str, gpu: int, max_q: int, item_ids, shard: int=0, num_shards: int=1, checkpoints=CHECKPOINTS, preserve: bool=True, seed: int=0) -> dict:
    from src.data_prep.canonical import load_manifest
    from src.inference.engine import build_item, fresh_decode, load_adapter
    from src.attacks.official_pv import official_pv_fn_factory
    from .triples import TRIPLES
    from .verifier import Verifier
    tri = TRIPLES[triple_key]
    dev = f'cuda:{gpu}'
    base_ad, base_pp, _ = load_adapter(tri['base'], gpu)
    ver_ad, _, _ = load_adapter(tri['verifier'], gpu)
    V = Verifier(ver_ad)
    manifest = {r['item_id']: r for r in load_manifest(tri['base'])}
    from src.verifier_ladder.data import load_cell
    cell = load_cell(tri['base'], tri['set'])
    lab = {i: int(y) for i, y in zip(cell['ids'], cell['y'])}
    OUT.mkdir(parents=True, exist_ok=True)
    mine = item_ids[shard::num_shards]
    tagp = '' if preserve else '_nopres'
    stag = f'_seed{seed}' if seed else ''
    out_path = OUT / f'{triple_key}_query{tagp}{stag}_s{shard}.jsonl'
    import glob as _glob
    done = set()
    for _f in _glob.glob(str(OUT / f'{triple_key}_query{tagp}{stag}_s*.jsonl')):
        for line in open(_f):
            if line.strip():
                try:
                    _r = json.loads(line)
                    if 'item_id' in _r:
                        done.add(_r['item_id'])
                except Exception:
                    pass
    todo = [i for i in mine if i not in done]
    print(f'[{triple_key} query s{shard}] todo={len(todo)} max_q={max_q}', flush=True)
    fh = open(out_path, 'a')
    for iid in todo:
        row = manifest[iid]
        q, ans = (row['question'], row['committed_clean_answer'])
        y = lab[iid]
        s_dir = 1.0 if y == 1 else -1.0
        try:
            ctx = build_item(base_ad, base_pp, row, dev)
            off = official_pv_fn_factory(base_ad, row)
            committed = ctx.row['committed_clean_answer']
            x0 = ctx.raw01.detach().to(torch.float32)
            clean_u8 = _u8(x0)

            def ver_score_u8(u8):
                img = Image.fromarray(np.transpose(u8, (1, 2, 0)), 'RGB')
                return V.yes_prob(img, q, ans)['score']

            def preserved(u8):
                text, _ = fresh_decode(base_ad, ctx, off(torch.from_numpy(u8)).to(dev))
                return text == committed
            g = torch.Generator(device=dev)
            g.manual_seed(SEED + seed * 1000003 + hash(iid) % 10000)
            x = x0.clone()
            last_ok_u8 = clean_u8
            ckpt_scores = {}
            queries = 0
            ck = sorted((c for c in checkpoints if c <= max_q))
            ci = 0
            while queries < max_q:
                delta = torch.randint(0, 2, x0.shape, generator=g, device=dev, dtype=torch.float32) * 2 - 1
                xp = project(x + SPSA_C * delta, x0, EPS)
                xm = project(x - SPSA_C * delta, x0, EPS)
                sp = ver_score_u8(_u8(xp))
                sm = ver_score_u8(_u8(xm))
                queries += 2
                ghat = s_dir * (sp - sm) / (2 * SPSA_C) * delta
                with torch.no_grad():
                    x = project(x - SPSA_LR * torch.nan_to_num(ghat).sign(), x0, EPS).detach()
                hit_ckpt = ci < len(ck) and queries >= ck[ci]
                niter = queries // 2
                if not preserve:
                    last_ok_u8 = _u8(x)
                elif hit_ckpt or niter % PRES_EVERY == 0:
                    cand = _u8(x)
                    if preserved(cand):
                        last_ok_u8 = cand
                while ci < len(ck) and queries >= ck[ci]:
                    ckpt_scores[str(ck[ci])] = ver_score_u8(last_ok_u8)
                    ci += 1
            for c in ck[ci:]:
                ckpt_scores[str(c)] = ver_score_u8(last_ok_u8)
            gl = int(np.abs(last_ok_u8.astype(np.int32) - clean_u8.astype(np.int32)).max())
            fh.write(json.dumps({'item_id': iid, 'label': y, 'checkpoint_scores': ckpt_scores, 'final_linf_greylevels': gl, 'used_clean': bool(np.array_equal(last_ok_u8, clean_u8))}) + '\n')
            fh.flush()
            print(f'  [{triple_key} query s{shard}] {iid} {ckpt_scores}', flush=True)
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
    for f in sorted(glob.glob(str(OUT / f'{triple_key}_query{tagp}_s*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if 'ERROR' not in r:
                    rows.append(r)
    by = {r['item_id']: r for r in rows}
    y = np.array([r['label'] for r in by.values()])
    cks = sorted({int(c) for r in by.values() for c in r['checkpoint_scores']}, key=int)
    out = {'triple': triple_key, 'n': len(y), 'budgets': {}}
    for c in cks:
        s = np.array([r['checkpoint_scores'].get(str(c), list(r['checkpoint_scores'].values())[-1]) for r in by.values()])
        out['budgets'][str(c)] = round(auroc(y, s), 4)
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--triple', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max-q', type=int, default=5000)
    ap.add_argument('--items', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--no-preserve', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.analyze:
        print(json.dumps(analyze(a.triple + ('' if not a.no_preserve else '')), indent=2))
    else:
        ids = json.load(open(a.items))['item_ids']
        print(json.dumps(run(a.triple, a.gpu, a.max_q, ids, a.shard, a.num_shards, preserve=not a.no_preserve, seed=a.seed), indent=2))
