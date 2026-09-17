from __future__ import annotations
import argparse
import json
import math
import numpy as np
from src.config import DATA_DIR, MANIFESTS_DIR
from src.inference.answer_format import answer_spec
from src.inference.models.factory import build_adapter
from src.data_prep.datasets.base import DatasetItem
from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
SCORE_MAX_SIDE = 336

def _ds(img):
    w, h = img.size
    if max(w, h) <= SCORE_MAX_SIDE:
        return img
    s = SCORE_MAX_SIDE / max(w, h)
    return img.resize((int(w * s), int(h * s)))
FINAL = MANIFESTS_DIR / 'final'
DRAW = DATA_DIR / 'draws'
OUT = DATA_DIR / 'pipeline'
DATASET_OF = {'gqa_train_probe_train': 'gqa', 'gqa_train_probe_val': 'gqa', 'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqav2', 'pope_ood': 'pope'}
DRAW_OF = {'gqa_train_probe_train': 'gqa_train_probe', 'gqa_train_probe_val': 'gqa_train_probe', 'gqa_val_eval': 'gqa_val_eval', 'vqav2_ood': 'vqav2_ood', 'pope_ood': 'pope_ood'}

def _geomean(probs):
    return round(math.exp(sum((math.log(max(p, 1e-12)) for p in probs)) / len(probs)), 6) if probs else None

def run(model, setname, gpu=0, shard=0, num_shards=1):
    dataset = DATASET_OF[setname]
    out_dir = OUT / model / setname
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f'shard_{shard:02d}.npz'
    jsonl_path = out_dir / f'shard_{shard:02d}.jsonl'
    if npz_path.exists() and jsonl_path.exists():
        print(f'skip {model}/{setname}/{shard} (done)', flush=True)
        return
    adapter = build_adapter(model, device=f'cuda:{gpu}')
    adapter.load()
    ds_adapter = get_ds_adapter(dataset)
    manifest = [json.loads(l) for l in open(FINAL / f'{setname}.jsonl') if l.strip()]
    draw = {json.loads(l)['item_id']: json.loads(l) for l in open(DRAW / f'{DRAW_OF[setname]}.jsonl') if l.strip()}
    mine = manifest[shard::num_shards]
    ids, saplma, pik = ([], [], [])
    recs = []
    for r in mine:
        iid = r['item_id']
        row = draw.get(iid)
        if not row:
            continue
        it = DatasetItem.from_dict(row)
        spec = answer_spec(dataset, r['answer_type'])
        try:
            img = _ds(ds_adapter.get_image(it).convert('RGB'))
            ans = adapter.generate_answer(img, it.question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
            if not ans.token_ids:
                continue
            p, s = adapter.answer_hidden_pik_saplma(img, it.question, ans.token_ids, format_instruction=spec.format_instruction)
            if p is None or s is None:
                continue
            ids.append(iid)
            saplma.append(s.astype(np.float16))
            pik.append(p.astype(np.float16))
            recs.append({'item_id': iid, 'dataset': dataset, 'set': setname, 'answer_type': r['answer_type'], 'question': it.question, 'gold': it.answer, 'answer': ans.answer_text, 'token_probs': [round(x, 6) for x in ans.token_probs], 'tokprob_geomean': _geomean(ans.token_probs), 'n_ans_tokens': len(ans.token_ids)})
        except Exception as e:
            print(f'err {iid}: {str(e)[:80]}', flush=True)
    if ids:
        np.savez(npz_path, ids=np.array(ids), saplma=np.stack(saplma), pik=np.stack(pik))
        with open(jsonl_path, 'w') as f:
            for rec in recs:
                f.write(json.dumps(rec) + '\n')
    print(json.dumps({'model': model, 'set': setname, 'shard': shard, 'n': len(ids)}), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--set', required=True, dest='setname')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    a = ap.parse_args()
    run(a.model, a.setname, a.gpu, a.shard, a.num_shards)
