from __future__ import annotations
import glob
import hashlib
import json
from src.config import DATA_DIR, REPO_ROOT
from src.inference.answer_format import answer_spec
from src.inference.families import MODELS, SETS
ICLR = DATA_DIR / 'iclr2027'
MANIFEST_DIR = ICLR / 'manifest'
CHANNELS = DATA_DIR / 'adv' / 'channels'
GENERALITY = DATA_DIR / 'adv' / 'generality'
SHIPPED_POOLS = REPO_ROOT / 'artifacts' / 'pools'
DATASET_OF = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqav2', 'pope_ood': 'pope'}

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_per_item(model: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(CHANNELS / model / 'per_item_s*.jsonl'))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r['item_id']] = r
    return out

def load_draws() -> dict:
    return {s: {json.loads(l)['item_id']: json.loads(l) for l in open(DATA_DIR / 'draws' / f'{s}.jsonl') if l.strip()} for s in SETS}

def build(model: str, draws: dict, verify_hashes: bool=True) -> dict:
    pool = json.load(open(GENERALITY / model / 'pool.json'))['pool']
    shipped = json.load(open(SHIPPED_POOLS / f'{model}__pool.json'))['pool']
    pool_ids = [p['item_id'] for p in pool]
    shipped_ids = [p['item_id'] for p in shipped]
    if sorted(pool_ids) != sorted(shipped_ids):
        raise SystemExit(f'{model}: dev pool and shipped pool disagree, refusing to proceed')
    per_item = load_per_item(model)
    rows = []
    for p in pool:
        iid = p['item_id']
        setname = p['dataset']
        pi = per_item.get(iid)
        if pi is None:
            raise SystemExit(f'{model}/{iid}: missing from the submitted per-item store')
        draw = draws[setname].get(iid)
        if draw is None:
            raise SystemExit(f'{model}/{iid}: missing from data/draws/{setname}.jsonl')
        dataset = DATASET_OF[setname]
        spec = answer_spec(dataset, draw.get('answer_type'))
        png = DATA_DIR / pi['clean_png']
        if not png.exists():
            raise SystemExit(f'{model}/{iid}: clean png missing at {png}')
        rows.append({'item_id': iid, 'model': model, 'dataset': setname, 'dataset_key': dataset, 'group': p['group'], 'label': int(p['label']), 'gold': pi.get('gold'), 'question': pi['question'], 'answer_type': draw.get('answer_type'), 'format_key': spec.format_key, 'format_instruction': spec.format_instruction, 'max_new_tokens': spec.max_new_tokens, 'committed_clean_answer': pi['clean_answer'], 'committed_clean_tokprob': pi['clean_tokprob'], 'committed_direction': pi['direction'], 'clean_png': pi['clean_png'], 'clean_png_sha256': sha256_file(png) if verify_hashes else None})
    return {'model': model, 'n': len(rows), 'rows': rows}

def manifest_path(model: str):
    return MANIFEST_DIR / f'{model}.jsonl'

def load_manifest(model: str) -> list:
    return [json.loads(l) for l in open(manifest_path(model)) if l.strip()]

def cell_items(model: str, setname: str) -> list:
    return [r for r in load_manifest(model) if r['dataset'] == setname]

def main(verify_hashes: bool=True):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    draws = load_draws()
    summary = {}
    for model in MODELS:
        built = build(model, draws, verify_hashes)
        rows = built['rows']
        with open(manifest_path(model), 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        per_set = {}
        for s in SETS:
            rr = [r for r in rows if r['dataset'] == s]
            per_set[s] = {'n': len(rr), 'correct': sum((1 for r in rr if r['label'] == 1)), 'incorrect': sum((1 for r in rr if r['label'] == 0))}
            if per_set[s] != {'n': 1000, 'correct': 500, 'incorrect': 500}:
                raise SystemExit(f'{model}/{s}: pool is not 1000 items at 500/500: {per_set[s]}')
        summary[model] = {'n': len(rows), 'per_set': per_set, 'manifest_sha256': sha256_file(manifest_path(model))}
        print(f'{model:22s} n={len(rows)} ' + ' '.join((f"{s}:{per_set[s]['correct']}/{per_set[s]['incorrect']}" for s in SETS)), flush=True)
    json.dump(summary, open(MANIFEST_DIR / 'summary.json', 'w'), indent=2)
    print('wrote', MANIFEST_DIR / 'summary.json')
    return summary
if __name__ == '__main__':
    main()
