from __future__ import annotations
import argparse
import json
import math
import random
import numpy as np
import torch
from src.config import DATA_DIR
from src.estimators.channels import answer_spec
from src.hidden_states.common import _ds
from src.estimators.train_probe import SAPLMANet
from src.inference.models.factory import build_adapter
from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
PIPE = DATA_DIR / 'pipeline'
OUT = DATA_DIR / 'head_cache' / 'clean'
SETS = {'gqa_train': ('gqa', 'train', 8000), 'gqa_val': ('gqa', 'val', 2000), 'vqav2': ('vqav2', 'val', 2000), 'pope': ('pope', 'test', 2000)}
SEED = 23

def _geomean(ps):
    return round(math.exp(sum((math.log(max(p, 1e-12)) for p in ps)) / len(ps)), 6) if ps else None

def _probe(model, key, dev):
    meta = json.load(open(PIPE / model / 'probes_meta.json'))['channels'][key]
    net = SAPLMANet(meta['input_dim']).to(dev)
    net.load_state_dict(torch.load(PIPE / model / 'probe_weights' / f'{key}.pth', map_location=dev))
    net.eval()
    return (net, bool(meta['swapped']))

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
    sap_net, sap_sw = _probe(model, 'saplma', dev)
    pik_net, pik_sw = _probe(model, 'pik', dev)
    ad = get_ds_adapter(ds)
    items = ad.load_split(split)
    random.Random(SEED).shuffle(items)
    items = items[:N][shard::num_shards]

    def pc(net, sw, h):
        p = float(torch.sigmoid(net(torch.tensor(h.astype(np.float32), device=dev))).item())
        return round(1 - p if sw else p, 5)
    recs, sap_H, pik_H, ids = ([], [], [], [])
    for it in items:
        spec = answer_spec(ds, it.answer_type)
        try:
            img = _ds(ad.get_image(it).convert('RGB'))
            ans = adapter.generate_answer(img, it.question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
            if not ans.token_ids:
                continue
            vb = adapter.verbalized_confidence(img, it.question, ans.answer_text, format_instruction=spec.format_instruction)
            pik_h, sap_h = adapter.answer_hidden_pik_saplma(img, it.question, ans.token_ids, format_instruction=spec.format_instruction)
            if sap_h is None or pik_h is None:
                continue
            recs.append({'item_id': it.item_id, 'set': setname, 'dataset': ds, 'question': it.question, 'gold': it.answer, 'answer': ans.answer_text, 'tokprob': _geomean(ans.token_probs), 'verbalized': vb.value, 'saplma': pc(sap_net, sap_sw, sap_h), 'pik': pc(pik_net, pik_sw, pik_h), 'answer_type': it.answer_type, 'n_ans_tokens': len(ans.token_ids)})
            ids.append(it.item_id)
            sap_H.append(sap_h.astype(np.float16))
            pik_H.append(pik_h.astype(np.float16))
        except Exception as e:
            print(f'err {it.item_id}: {str(e)[:70]}', flush=True)
    with open(jpath, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    np.savez_compressed(npath, ids=np.array(ids), saplma=np.stack(sap_H) if sap_H else np.zeros((0, 1), np.float16), pik=np.stack(pik_H) if pik_H else np.zeros((0, 1), np.float16))
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
