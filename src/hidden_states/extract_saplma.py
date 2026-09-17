from __future__ import annotations
import argparse
import glob
import json
import numpy as np
from src.config import DATA_DIR, MANIFESTS_DIR
from src.estimators.channels import answer_spec
from src.hidden_states.common import _ds
from src.hidden_states.common import score_correct
from src.inference.models.factory import build_adapter
from src.data_prep.datasets.base import DatasetItem
from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
FINAL = MANIFESTS_DIR / 'final'
CACHE008 = DATA_DIR / 'faith' / 'cache'
OUT_ROOT = DATA_DIR / 'saplma' / 'hidden'
DRAW = DATA_DIR / 'draws'
DATASET_OF = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqav2', 'pope_ood': 'pope', 'gqa_train_probe_train': 'gqa', 'gqa_train_probe_val': 'gqa'}
PROBE_SETS = {'gqa_train_probe_train', 'gqa_train_probe_val'}
DRAW_OF = {'gqa_train_probe_train': 'gqa_train_probe', 'gqa_train_probe_val': 'gqa_train_probe'}

def _cache008_answers(model, dataset):
    out = {}
    for f in glob.glob(str(CACHE008 / model / 'shard_*.jsonl')):
        for l in open(f):
            l = l.strip()
            if l:
                r = json.loads(l)
                if r.get('dataset') == dataset and 'orig' in r and ('error' not in r):
                    out[r['item_id']] = (r['orig']['answer'], bool(r['orig']['correct']))
    return out

def run(model_key, setname, gpu=0, shard=0, num_shards=1):
    dataset = DATASET_OF[setname]
    out_dir = OUT_ROOT / model_key / setname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'shard_{shard:02d}.npz'
    if out_file.exists():
        print(f'skip {model_key}/{setname}/{shard} (done)', flush=True)
        return
    adapter = build_adapter(model_key, device=f'cuda:{gpu}')
    adapter.load()
    ds_adapter = get_ds_adapter(dataset)
    manifest = [json.loads(l) for l in open(FINAL / f'{setname}.jsonl') if l.strip()]
    draw = {json.loads(l)['item_id']: json.loads(l) for l in open(DRAW / f'{DRAW_OF.get(setname, setname)}.jsonl') if l.strip()}
    is_probe = setname in PROBE_SETS
    cache = None if is_probe else _cache008_answers(model_key, dataset)
    mine = manifest[shard::num_shards]
    hid, ids, labels = ([], [], [])
    for r in mine:
        iid = r['item_id']
        row = draw.get(iid)
        if not row:
            continue
        it = DatasetItem.from_dict(row)
        spec = answer_spec(dataset, r['answer_type'])
        try:
            img = _ds(ds_adapter.get_image(it).convert('RGB'))
            if is_probe:
                ans = adapter.generate_answer(img, it.question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
                answer_text = ans.answer_text
                label = score_correct(it.question, it.answer, answer_text)
            else:
                got = cache.get(iid)
                if not got:
                    continue
                answer_text, label = got
            h = adapter.answer_hidden_state(img, it.question, answer_text, format_instruction=spec.format_instruction)
            if h is None:
                continue
            hid.append(h.astype(np.float16))
            ids.append(iid)
            labels.append(1 if label else 0)
        except Exception as e:
            print(f'err {iid}: {str(e)[:80]}', flush=True)
    if hid:
        np.savez(out_file, hidden=np.stack(hid), ids=np.array(ids), labels=np.array(labels, dtype=np.int8))
    print(json.dumps({'model': model_key, 'set': setname, 'shard': shard, 'n': len(hid)}), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--set', required=True, dest='setname')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    a = ap.parse_args()
    run(a.model, a.setname, a.gpu, a.shard, a.num_shards)
