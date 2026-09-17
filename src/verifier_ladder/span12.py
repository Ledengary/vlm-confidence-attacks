from __future__ import annotations
import ast
import json
import numpy as np
from src.config import DATA_DIR, SEED
OUT = DATA_DIR / 'snowglobe' / 'verifier_escape'
DRAWS = DATA_DIR / 'draws'
MSHORT = {'internvl3_5_2b': 'internvl', 'gemma3_12b': 'gemma', 'molmo2_4b': 'molmo', 'llava_onevision_7b': 'llava'}
SSHORT = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqa', 'pope_ood': 'pope'}
SPAN12 = [('sp0', 'T0', 'llava_onevision_7b', 'gemma3_12b', 'vqav2_ood'), ('sp1', 'T1', 'llava_onevision_7b', 'internvl3_5_2b', 'vqav2_ood'), ('sp2', 'T2', 'gemma3_12b', 'llava_onevision_7b', 'gqa_val_eval'), ('sp3', 'T3', 'molmo2_4b', 'gemma3_12b', 'pope_ood'), ('sp4', 'T4', 'internvl3_5_2b', 'llava_onevision_7b', 'gqa_val_eval'), ('sp5', 'T5', 'internvl3_5_2b', 'molmo2_4b', 'pope_ood'), ('sp6', 'T6', 'gemma3_12b', 'molmo2_4b', 'vqav2_ood'), ('sp7', 'T7', 'molmo2_4b', 'internvl3_5_2b', 'gqa_val_eval'), ('sp8', 'T8', 'llava_onevision_7b', 'molmo2_4b', 'pope_ood'), ('sp9', 'T9', 'gemma3_12b', 'internvl3_5_2b', 'pope_ood'), ('sp10', 'T10', 'internvl3_5_2b', 'gemma3_12b', 'vqav2_ood'), ('sp11', 'T11', 'molmo2_4b', 'llava_onevision_7b', 'vqav2_ood')]

def _srcimg(setname, d):
    if setname == 'gqa_val_eval':
        return str(d['image_key'])
    m = d.get('meta')
    if isinstance(m, str):
        m = ast.literal_eval(m)
    return str((m or {}).get('coco_image_id', d['image_key']))

def common_pool(base, setname, per_class=100):
    p = OUT / f'sp_pool_{MSHORT[base]}_{SSHORT[setname]}.json'
    if p.exists():
        return json.load(open(p))['item_ids']
    from src.data_prep.canonical import load_manifest
    man = {r['item_id']: r for r in load_manifest(base) if r['dataset'] == setname}
    draws = {}
    for l in open(DRAWS / f'{setname}.jsonl'):
        if l.strip():
            r = json.loads(l)
            draws[r['item_id']] = r
    rng = np.random.default_rng(SEED)
    ordered = list(man.keys())
    ordered = [ordered[i] for i in rng.permutation(len(ordered))]
    picked = {1: [], 0: []}
    used = set()
    for iid in ordered:
        if iid not in draws:
            continue
        lab = int(man[iid]['label'])
        if len(picked[lab]) >= per_class:
            continue
        img = _srcimg(setname, draws[iid])
        if img in used:
            continue
        picked[lab].append(iid)
        used.add(img)
        if len(picked[1]) >= per_class and len(picked[0]) >= per_class:
            break
    ids = picked[1] + picked[0]
    json.dump({'item_ids': ids, 'n_correct': len(picked[1]), 'n_wrong': len(picked[0]), 'base': base, 'set': setname, 'rule': 'first-100-correct + first-100-incorrect, seed-23 canonical permutation, image-disjoint'}, open(p, 'w'), indent=2)
    return ids

def full_pool(base, setname):
    p = OUT / f'sp_full_{MSHORT[base]}_{SSHORT[setname]}.json'
    if p.exists():
        return json.load(open(p))['item_ids']
    from src.data_prep.canonical import load_manifest
    ids = [r['item_id'] for r in load_manifest(base) if r['dataset'] == setname]
    json.dump({'item_ids': ids, 'n': len(ids), 'base': base, 'set': setname}, open(p, 'w'), indent=2)
    return ids

def stim_dir(base, setname):
    return f'data/snowglobe/verifier_escape/matrix_images/{MSHORT[base]}_{SSHORT[setname]}'

def build_triples_json():
    entries = {}
    if (OUT / 'matrix_triples.json').exists():
        entries = json.load(open(OUT / 'matrix_triples.json'))
    for spkey, lab, b, v, s in SPAN12:
        common_pool(b, s)
        full_pool(b, s)
        entries[spkey] = {'base': b, 'verifier': v, 'set': s, 'label': lab, 'items': f'data/snowglobe/verifier_escape/sp_pool_{MSHORT[b]}_{SSHORT[s]}.json', 'full_items': f'data/snowglobe/verifier_escape/sp_full_{MSHORT[b]}_{SSHORT[s]}.json', 'base_images_dir': stim_dir(b, s), 'full_images_dir': f'data/snowglobe/verifier_escape/matrix_full_images/{MSHORT[b]}_{SSHORT[s]}'}
    json.dump(entries, open(OUT / 'matrix_triples.json', 'w'), indent=2)
    return entries
if __name__ == '__main__':
    e = build_triples_json()
    print(f"span12: wrote {len([k for k in e if k.startswith('sp')])} sp entries")
    for spkey, lab, b, v, s in SPAN12:
        ids = common_pool(b, s)
        d = json.load(open(OUT / f'sp_pool_{MSHORT[b]}_{SSHORT[s]}.json'))
        print(f"  {spkey}({lab}) {MSHORT[b]}->{MSHORT[v]} ({SSHORT[s]}): pool {len(ids)} ({d['n_correct']}c/{d['n_wrong']}w)")
