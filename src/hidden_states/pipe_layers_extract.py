from __future__ import annotations
import argparse
import glob
import json
import random
import numpy as np
import torch
from src.config import DATA_DIR
from src.estimators.channels import answer_spec
from src.hidden_states.common import _ds
from src.hidden_states.common import _score_inputs
from src.inference.models.factory import build_adapter
from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
OUT = DATA_DIR / 'head_cache' / 'layers'
SETS = {'gqa_train': ('gqa', 'train', 4000), 'gqa_val': ('gqa', 'val', 1500), 'vqav2': ('vqav2', 'val', 1500), 'pope': ('pope', 'test', 1500)}
SEED = 23

@torch.no_grad()
def _all_layer_both_loci(adapter, image, question, ans_ids, spec):
    inputs2, L = _score_inputs(adapter, image, question, ans_ids, spec)
    out = adapter.model(**inputs2, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    saplma = torch.stack([h[0, -1] for h in hs]).to(torch.float16).cpu().numpy()
    pik = torch.stack([h[0, L - 1] for h in hs]).to(torch.float16).cpu().numpy()
    return (saplma, pik)

def run(model, setname, gpu=0, shard=0, num_shards=1):
    dev = f'cuda:{gpu}'
    ds, split, N = SETS[setname]
    out_dir = OUT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f'{setname}_shard_{shard:02d}.jsonl'
    npath = out_dir / f'{setname}_shard_{shard:02d}.npz'
    if jpath.exists() and npath.exists():
        print('skip (done)')
        return
    adapter = build_adapter(model, device=dev)
    adapter.load()
    ad = get_ds_adapter(ds)
    items = ad.load_split(split)
    random.Random(SEED).shuffle(items)
    items = items[:N][shard::num_shards]
    recs, SAP, PIK, ids = ([], [], [], [])
    for it in items:
        spec = answer_spec(ds, it.answer_type)
        try:
            img = _ds(ad.get_image(it).convert('RGB'))
            ans = adapter.generate_answer(img, it.question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
            if not ans.token_ids:
                continue
            sap, pik = _all_layer_both_loci(adapter, img, it.question, ans.token_ids, spec)
            recs.append({'item_id': it.item_id, 'set': setname, 'dataset': ds, 'gold': it.answer, 'answer': ans.answer_text, 'answer_type': it.answer_type})
            ids.append(it.item_id)
            SAP.append(sap)
            PIK.append(pik)
        except Exception as e:
            print(f'err {it.item_id}: {str(e)[:70]}', flush=True)
    with open(jpath, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    np.savez_compressed(npath, ids=np.array(ids), saplma=np.stack(SAP) if SAP else np.zeros((0, 1, 1), np.float16), pik=np.stack(PIK) if PIK else np.zeros((0, 1, 1), np.float16))
    print(json.dumps({'model': model, 'set': setname, 'shard': shard, 'n': len(recs)}), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--set', required=True, dest='setname')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    a = ap.parse_args()
    run(a.model, a.setname, a.gpu, a.shard, a.num_shards)
