from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.config import DATA_DIR
from src.verifier_ladder.verifier import Verifier
OUT = DATA_DIR / 'snowglobe' / 'verifier_escape'

def base_answers_labels(base, setname):
    from src.data_prep.canonical import load_manifest
    from src.verifier_ladder.data import load_cell
    man = {r['item_id']: r for r in load_manifest(base) if r['dataset'] == setname}
    cell = load_cell(base, setname)
    lab = {i: int(y) for i, y in zip(cell['ids'], cell['y'])}
    return (man, lab)

def run(triple_key, gpu, full=False, shard=0, num_shards=1):
    from src.inference.engine import load_adapter
    from src.verifier_ladder.triples import TRIPLES
    tri = TRIPLES[triple_key]
    man, lab = base_answers_labels(tri['base'], tri['set'])
    items = json.load(open(tri['full_items' if full else 'items']))['item_ids']
    img_dir = Path(tri['full_images_dir' if full else 'base_images_dir'])
    tag = 'spscoresF' if full else 'spscores'
    out_path = OUT / f'{triple_key}_{tag}_s{shard}.jsonl'
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                done.add(json.loads(l)['item_id'])
    adapter, _, _ = load_adapter(tri['verifier'], gpu)
    V = Verifier(adapter)
    mine = [i for i in items[shard::num_shards] if i in lab and i not in done]
    print(f'[{triple_key} {tag} s{shard}] todo={len(mine)}', flush=True)
    fh = open(out_path, 'a')
    for n, iid in enumerate(mine):
        r = man[iid]
        q, ans = (r['question'], r['committed_clean_answer'])
        rec = {'item_id': iid, 'label': lab[iid]}
        cln = img_dir / f'{iid}__clean.png'
        rec['c0_clean'] = V.score_image_file(cln, q, ans)['score'] if cln.exists() else None
        for d in ('min', 'max'):
            p = img_dir / f'{iid}__{d}__adv.png'
            rec[f'c1_{d}'] = V.score_image_file(p, q, ans)['score'] if p.exists() else None
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        if (n + 1) % 50 == 0:
            print(f'  [{triple_key} {tag} s{shard}] {n + 1}/{len(mine)}', flush=True)
    fh.close()
    return {'scored': len(mine)}
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--triple', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--common', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(run(a.triple, a.gpu, a.full, a.shard, a.num_shards), indent=2))
