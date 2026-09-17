from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np
import torch
from src.config import DATA_DIR
from src.estimators.channels import answer_spec
from src.hidden_states.common import _ds
from src.estimators.coupled_readout import _score_inputs, image_std_mean
from src.estimators.coupled_readout import Subjects3, _run_pgd, _margin, _greedy
from src.inference.models.factory import build_adapter
from src.data_prep.datasets.base import DatasetItem
from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
OUT = DATA_DIR / 'coupled'
SEED = 23
EPS = 8
DATASETS = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqav2', 'pope_ood': 'pope'}
TEST_SPLITS = ['gqa_val_eval', 'vqav2_ood', 'pope_ood']
MANIFEST = {'train': 'manifests/final/gqa_train_probe_train.jsonl', 'val': 'manifests/final/gqa_train_probe_val.jsonl'}
KCOMP = 16

def ccps_split_for(split):
    return {'train': 'train', 'val': 'val'}.get(split, 'test')

def load_ccps_seqs(model, split):
    sub = ccps_split_for(split)
    tmp = {}
    for f in glob.glob(str(DATA_DIR / 'ccps' / 'features' / model / sub / '*.jsonl')):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            tmp.setdefault(str(r['item_id']), []).append((int(r['token_idx']), int(r['token_id'])))
    return {k: [t for _, t in sorted(v)] for k, v in tmp.items()}

def build_pool(model, split):
    labj = {}
    for l in open(DATA_DIR / 'pipeline' / model / 'judged.jsonl'):
        if l.strip():
            r = json.loads(l)
            labj[str(r['item_id'])] = int(r['correct'])
    if split in ('train', 'val'):
        ids = [json.loads(l)['item_id'] for l in open(MANIFEST[split]) if l.strip()]
        draws = {json.loads(l)['item_id']: json.loads(l) for l in open(DATA_DIR / 'draws' / 'gqa_train_probe.jsonl') if l.strip()}
        pool = []
        for iid in ids:
            if iid in labj and iid in draws:
                row = draws[iid]
                pool.append({'item_id': iid, 'dataset': 'gqa_train_probe', 'ds': 'gqa', 'group': 'correct' if labj[iid] == 1 else 'wrong', 'label': labj[iid], 'gold': row.get('answer'), 'row': row})
        return pool
    poolj = json.load(open(DATA_DIR / 'adv' / 'generality' / model / 'pool.json'))['pool']
    draws = {json.loads(l)['item_id']: json.loads(l) for l in open(DATA_DIR / 'draws' / f'{split}.jsonl') if l.strip()}
    pool = []
    for p in poolj:
        if p['dataset'] != split:
            continue
        iid = p['item_id']
        if iid not in draws:
            continue
        pool.append({'item_id': iid, 'dataset': split, 'ds': DATASETS[split], 'group': p['group'], 'label': int(p['label']), 'gold': p.get('gold'), 'row': draws[iid]})
    return pool

def _hid32(hs_last, L):
    return (hs_last[L - 1].float().cpu().numpy().astype(np.float32), hs_last[-1].float().cpu().numpy().astype(np.float32))

def run(model, split, gpu=0, shard=0, num_shards=8):
    torch.set_num_threads(1)
    is_test = split in TEST_SPLITS
    d = OUT / model
    d.mkdir(parents=True, exist_ok=True)
    pool = build_pool(model, split)
    mine = pool[shard::num_shards]
    meta_p = d / f'{split}_meta_s{shard}.jsonl'
    npz_p = d / f'{split}_hidden_s{shard}.npz'
    n_npz = 0
    if npz_p.exists():
        n_npz = int(np.load(npz_p, allow_pickle=True)['ids'].shape[0])
    done = set()
    if meta_p.exists():
        lines = [l for l in open(meta_p) if l.strip()]
        if len(lines) != n_npz:
            lines = lines[:n_npz]
            with open(meta_p, 'w') as fh:
                for l in lines:
                    fh.write(l if l.endswith('\n') else l + '\n')
        done = {json.loads(l)['item_id'] for l in lines}
    todo = [p for p in mine if p['item_id'] not in done]
    print(f'[{model}/{split} s{shard}/{num_shards} gpu{gpu}] mine={len(mine)} done={len(done)} todo={len(todo)} test={is_test}', flush=True)
    if not todo:
        return
    dev = f'cuda:{gpu}'
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    adapter = build_adapter(model, device=dev)
    adapter.load()
    for p_ in adapter.model.parameters():
        p_.requires_grad_(False)
    subj = Subjects3(model, dev)
    dim = subj.dim
    std = image_std_mean(adapter)
    eps_pv = EPS / 255.0 / std
    alpha = eps_pv / 4
    ds_adapters = {}
    ccps = load_ccps_seqs(model, split)
    buf = {'ids': [], 'h_pik': [], 'h_sap': []}
    if is_test:
        buf.update({'h_pik_adv': [], 'h_sap_adv': []})
    if npz_p.exists():
        old = np.load(npz_p, allow_pickle=True)
        for k in buf:
            if k in old.files:
                buf[k] = list(old[k])
    fj = open(meta_p, 'a')
    n_ccps = n_gen = 0
    for i, p in enumerate(todo):
        iid = p['item_id']
        setname = p['dataset']
        ds = p['ds']
        if ds not in ds_adapters:
            ds_adapters[ds] = get_ds_adapter(ds)
        it = DatasetItem.from_dict(p['row'])
        spec = answer_spec(ds, p['row'].get('answer_type'))
        try:
            img = _ds(ds_adapters[ds].get_image(it).convert('RGB'))
            clean_answer = None
            ccps_ids = ccps.get(iid)
            if is_test:
                gen = adapter.generate_answer(img, it.question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
                ans_ids = list(gen.token_ids)
                clean_answer = gen.answer_text
                src = 'gen_test'
                if ccps_ids and ccps_ids == ans_ids:
                    src = 'ccps'
                    n_ccps += 1
                else:
                    n_gen += 1
            else:
                ans_ids = ccps_ids
                src = 'ccps'
                if not ans_ids:
                    gen = adapter.generate_answer(img, it.question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
                    ans_ids = list(gen.token_ids)
                    src = 'gen'
                    n_gen += 1
                else:
                    n_ccps += 1
            if not ans_ids:
                print(f'skip {iid}: no answer tokens', flush=True)
                continue
            inputs2, L = _score_inputs(adapter, img, it.question, ans_ids, spec)
            md = inputs2['pixel_values'].dtype
            pv0 = inputs2['pixel_values'].detach().float()
            with torch.no_grad():
                o0 = adapter.model(**{**inputs2, 'pixel_values': pv0.to(md)}, output_hidden_states=True, use_cache=False)
                logits0 = o0.logits[0].float()
                sc0 = {k: float(v) for k, v in subj.compute(logits0, L, ans_ids, o0.hidden_states[-1][0]).items()}
                cpik, csap = _hid32(o0.hidden_states[-1][0], L)
                drow = logits0[L - 1]
                dargmax = int(drow.argmax())
                order = torch.argsort(drow, descending=True).tolist()
                comp = [t for t in order if t != ans_ids[0]][:KCOMP]
                comp_logits = [float(drow[t]) for t in comp]
            rec = {'item_id': iid, 'dataset': setname, 'split': split, 'label': p['label'], 'group': p['group'], 'gold': p['gold'], 'n_ans': len(ans_ids), 'ans_ids': ans_ids, 'ans_src': src, 'ccps_match_first': ccps.get(iid, [None])[0] == ans_ids[0], 'decode_first_argmax': dargmax, 'argmax_eq_ans0': dargmax == ans_ids[0], 'clean': sc0, 'comp_ids': comp, 'comp_logits': [round(x, 5) for x in comp_logits]}
            del o0, logits0
            apik = asap = None
            if is_test:
                torch.cuda.empty_cache()
                for attempt in range(2):
                    try:
                        sgn = -1.0 if p['group'] == 'correct' else 1.0
                        best_delta, base, best_val, found, pres = _run_pgd(adapter, subj, inputs2, L, ans_ids, 'tokprob', sgn, pv0, md, eps_pv, alpha)
                        adv_pv = pv0 + best_delta
                        with torch.no_grad():
                            oa = adapter.model(**{**inputs2, 'pixel_values': adv_pv.to(md)}, output_hidden_states=True, use_cache=False)
                            la = oa.logits[0].float()
                            sca = {k: float(v) for k, v in subj.compute(la, L, ans_ids, oa.hidden_states[-1][0]).items()}
                            apik, asap = _hid32(oa.hidden_states[-1][0], L)
                        del oa, la
                        margin = _margin(adapter, inputs2, adv_pv, L, ans_ids, md)
                        adv_answer = _greedy(adapter, img, it.question, spec, adv_pv, md)
                        break
                    except RuntimeError as re_:
                        if 'out of memory' in str(re_).lower() and attempt == 0:
                            torch.cuda.empty_cache()
                            print(f'oom-retry {iid}', flush=True)
                            continue
                        raise
                bi = adv_answer == clean_answer
                rec.update({'adv': sca, 'clean_answer': clean_answer, 'adv_answer': None if bi else adv_answer, 'margin_min': round(margin, 5) if margin is not None else None, 'byte_identical': bi, 'found': found, 'pres_rate': round(pres, 4)})
            buf['ids'].append(iid)
            buf['h_pik'].append(cpik)
            buf['h_sap'].append(csap)
            if is_test:
                buf['h_pik_adv'].append(apik)
                buf['h_sap_adv'].append(asap)
            fj.write(json.dumps(rec) + '\n')
            fj.flush()
            if (i + 1) % 200 == 0:
                np.savez(npz_p, **{k: np.array(v) for k, v in buf.items()})
                torch.cuda.empty_cache()
                print(f'[{model}/{split} s{shard}] {i + 1}/{len(todo)} ccps={n_ccps} gen={n_gen}', flush=True)
        except Exception as e:
            print(f'err {iid}: {str(e)[:160]}', flush=True)
    np.savez(npz_p, **{k: np.array(v) for k, v in buf.items()})
    fj.close()
    print(json.dumps({'model': model, 'split': split, 'shard': shard, 'n_written': len(todo), 'ccps': n_ccps, 'gen': n_gen}), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=8)
    a = ap.parse_args()
    run(a.model, a.split, a.gpu, a.shard, a.num_shards)
