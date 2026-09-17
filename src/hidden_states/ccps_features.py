from __future__ import annotations
import argparse, glob, json, os
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from src.config import DATA_DIR
from src.estimators.channels import answer_spec
from src.hidden_states.common import _ds
from src.hidden_states.common import _score_inputs
from src.inference.models.factory import build_adapter
from src.data_prep.datasets.base import DatasetItem
from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
SEED = 23
PEI_RADIUS = 20.0
PEI_STEPS = 5
OUT = DATA_DIR / 'ccps' / 'features'
CH = DATA_DIR / 'adv' / 'channels'
DATASETS = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqav2', 'pope_ood': 'pope'}
FEATURE_COLUMNS = ['original_log_prob_actual', 'original_prob_actual', 'original_logit_actual', 'original_prob_argmax', 'original_logit_argmax', 'original_entropy', 'original_margin_logit_top1_top2', 'original_margin_prob_top1_top2', 'original_norm_logits_L2', 'original_std_logits', 'original_norm_hidden_state_L2', 'is_actual_token_original_argmax', 'jacobian_norm_token', 'epsilon_to_flip_token', 'pei_value_token', 'perturbed_log_prob_actual_mean', 'perturbed_log_prob_actual_std', 'perturbed_log_prob_actual_min', 'perturbed_log_prob_actual_max', 'perturbed_prob_actual_mean', 'perturbed_prob_actual_std', 'perturbed_prob_actual_min', 'perturbed_prob_actual_max', 'perturbed_logit_actual_mean', 'perturbed_logit_actual_std', 'perturbed_logit_actual_min', 'perturbed_logit_actual_max', 'delta_log_prob_actual_from_original_mean', 'delta_log_prob_actual_from_original_std', 'delta_log_prob_actual_from_original_min', 'delta_log_prob_actual_from_original_max', 'perturbed_prob_argmax_mean', 'perturbed_prob_argmax_std', 'perturbed_prob_argmax_min', 'perturbed_prob_argmax_max', 'perturbed_logit_argmax_mean', 'perturbed_logit_argmax_std', 'perturbed_logit_argmax_min', 'perturbed_logit_argmax_max', 'did_argmax_change_from_original_mean', 'did_argmax_change_from_original_std', 'did_argmax_change_from_original_min', 'did_argmax_change_from_original_max', 'perturbed_entropy_mean', 'perturbed_entropy_std', 'perturbed_entropy_min', 'perturbed_entropy_max', 'perturbed_margin_logit_top1_top2_mean', 'perturbed_margin_logit_top1_top2_std', 'perturbed_margin_logit_top1_top2_min', 'perturbed_margin_logit_top1_top2_max', 'perturbed_norm_logits_L2_mean', 'perturbed_norm_logits_L2_std', 'perturbed_norm_logits_L2_min', 'perturbed_norm_logits_L2_max', 'kl_div_perturbed_from_original_mean', 'kl_div_perturbed_from_original_std', 'kl_div_perturbed_from_original_min', 'kl_div_perturbed_from_original_max', 'js_div_perturbed_from_original_mean', 'js_div_perturbed_from_original_std', 'js_div_perturbed_from_original_min', 'js_div_perturbed_from_original_max', 'cosine_sim_logits_perturbed_to_original_mean', 'cosine_sim_logits_perturbed_to_original_std', 'cosine_sim_logits_perturbed_to_original_min', 'cosine_sim_logits_perturbed_to_original_max', 'cosine_sim_hidden_perturbed_to_original_mean', 'cosine_sim_hidden_perturbed_to_original_std', 'cosine_sim_hidden_perturbed_to_original_min', 'cosine_sim_hidden_perturbed_to_original_max', 'l2_dist_hidden_perturbed_from_original_mean', 'l2_dist_hidden_perturbed_from_original_std', 'l2_dist_hidden_perturbed_from_original_min', 'l2_dist_hidden_perturbed_from_original_max']

def compute_jacobians_batch_analytical(hidden_states, logits, token_ids, lm_head_weight):
    device = logits.device
    dtype = logits.dtype
    lm_head_weight = lm_head_weight.to(device=device, dtype=dtype)
    probs = F.softmax(logits, dim=-1)
    if probs.dtype != lm_head_weight.dtype:
        lm_head_weight = lm_head_weight.to(dtype=probs.dtype)
    weighted_sums = torch.matmul(probs, lm_head_weight)
    w_targets = lm_head_weight[token_ids]
    return weighted_sums - w_targets

def generate_perturbation_trajectories_batch(hidden_states, jacobians, pei_radius, pei_steps):
    T, D = hidden_states.shape
    jacobian_norms = torch.norm(jacobians, dim=1, keepdim=True)
    safe_norms = torch.where(jacobian_norms > 1e-09, jacobian_norms, torch.ones_like(jacobian_norms))
    jacobian_directions = jacobians / safe_norms
    zero_mask = (jacobian_norms <= 1e-09).squeeze(-1)
    jacobian_directions[zero_mask] = 0
    delta_r = pei_radius / pei_steps
    radii = torch.arange(1, pei_steps + 1, device=hidden_states.device, dtype=hidden_states.dtype) * delta_r
    return hidden_states.unsqueeze(1) + radii.view(1, -1, 1) * jacobian_directions.unsqueeze(1)

def get_perturbed_logits_batch(lm_head, perturbed_hidden):
    T, S, D = perturbed_hidden.shape
    with torch.no_grad():
        logits_flat = lm_head(perturbed_hidden.view(-1, D))
    return logits_flat.view(T, S, -1)

def extract_ccps_features_batch(hidden_states, original_logits, jacobians, perturbed_hidden_states, perturbed_logits, token_ids, pei_radius, pei_steps) -> List[Dict[str, float]]:
    T = hidden_states.shape[0]
    all_features = []
    probs_0 = F.softmax(original_logits, dim=-1)
    log_probs_0 = F.log_softmax(original_logits, dim=-1)
    top2_values, top2_indices = torch.topk(original_logits, 2, dim=-1)
    argmax_0 = top2_indices[:, 0]
    second_best_0 = top2_indices[:, 1]
    perturbed_probs = F.softmax(perturbed_logits, dim=-1)
    perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)
    delta_r = pei_radius / pei_steps
    for t in range(T):
        features = {}
        token_id = token_ids[t].item()
        features['original_log_prob_actual'] = log_probs_0[t, token_id].item()
        features['original_prob_actual'] = probs_0[t, token_id].item()
        features['original_logit_actual'] = original_logits[t, token_id].item()
        features['original_prob_argmax'] = probs_0[t, argmax_0[t]].item()
        features['original_logit_argmax'] = original_logits[t, argmax_0[t]].item()
        features['original_entropy'] = -torch.sum(probs_0[t] * torch.log(probs_0[t] + 1e-09)).item()
        features['original_margin_logit_top1_top2'] = (original_logits[t, argmax_0[t]] - original_logits[t, second_best_0[t]]).item()
        features['original_margin_prob_top1_top2'] = (probs_0[t, argmax_0[t]] - probs_0[t, second_best_0[t]]).item()
        features['original_norm_logits_L2'] = torch.norm(original_logits[t]).item()
        features['original_std_logits'] = torch.std(original_logits[t]).item()
        features['original_norm_hidden_state_L2'] = torch.norm(hidden_states[t]).item()
        features['is_actual_token_original_argmax'] = int(token_id == argmax_0[t].item())
        features['jacobian_norm_token'] = torch.norm(jacobians[t]).item()
        epsilon_to_flip = float('inf')
        perturbed_argmaxes = torch.argmax(perturbed_logits[t], dim=-1)
        for k in range(pei_steps):
            if perturbed_argmaxes[k].item() != argmax_0[t].item():
                epsilon_to_flip = (k + 1) * delta_r
                break
        features['epsilon_to_flip_token'] = epsilon_to_flip
        log_p_original = log_probs_0[t, token_id].item()
        f_values = [0.0]
        for k in range(pei_steps):
            f_values.append(max(0.0, log_p_original - perturbed_log_probs[t, k, token_id].item()))
        features['pei_value_token'] = sum(((f_values[k] + f_values[k + 1]) / 2.0 for k in range(pei_steps))) / pei_steps if pei_steps > 0 else 0.0
        pm = {k: [] for k in ('perturbed_log_prob_actual', 'perturbed_prob_actual', 'perturbed_logit_actual', 'delta_log_prob_actual_from_original', 'perturbed_prob_argmax', 'perturbed_logit_argmax', 'did_argmax_change_from_original', 'perturbed_entropy', 'perturbed_margin_logit_top1_top2', 'perturbed_norm_logits_L2', 'kl_div_perturbed_from_original', 'js_div_perturbed_from_original', 'cosine_sim_logits_perturbed_to_original', 'cosine_sim_hidden_perturbed_to_original', 'l2_dist_hidden_perturbed_from_original')}
        for s in range(pei_steps):
            pm['perturbed_log_prob_actual'].append(perturbed_log_probs[t, s, token_id].item())
            pm['perturbed_prob_actual'].append(perturbed_probs[t, s, token_id].item())
            pm['perturbed_logit_actual'].append(perturbed_logits[t, s, token_id].item())
            pm['delta_log_prob_actual_from_original'].append(log_probs_0[t, token_id].item() - perturbed_log_probs[t, s, token_id].item())
            argmax_p = perturbed_argmaxes[s].item()
            pm['perturbed_prob_argmax'].append(perturbed_probs[t, s, argmax_p].item())
            pm['perturbed_logit_argmax'].append(perturbed_logits[t, s, argmax_p].item())
            pm['did_argmax_change_from_original'].append(int(argmax_p != argmax_0[t].item()))
            pm['perturbed_entropy'].append((-torch.sum(perturbed_probs[t, s] * torch.log(perturbed_probs[t, s] + 1e-09))).item())
            top2_p = torch.topk(perturbed_logits[t, s], 2)
            pm['perturbed_margin_logit_top1_top2'].append((top2_p.values[0] - top2_p.values[1]).item() if len(top2_p.values) >= 2 else 0.0)
            pm['perturbed_norm_logits_L2'].append(torch.norm(perturbed_logits[t, s]).item())
            pm['kl_div_perturbed_from_original'].append(F.kl_div(perturbed_log_probs[t, s], probs_0[t], reduction='sum').item())
            m_probs = 0.5 * (probs_0[t] + perturbed_probs[t, s])
            pm['js_div_perturbed_from_original'].append((0.5 * F.kl_div(log_probs_0[t], m_probs, reduction='sum') + 0.5 * F.kl_div(perturbed_log_probs[t, s], m_probs, reduction='sum')).item())
            pm['cosine_sim_logits_perturbed_to_original'].append(F.cosine_similarity(original_logits[t].unsqueeze(0), perturbed_logits[t, s].unsqueeze(0)).item())
            pm['cosine_sim_hidden_perturbed_to_original'].append(F.cosine_similarity(hidden_states[t].unsqueeze(0), perturbed_hidden_states[t, s].unsqueeze(0)).item())
            pm['l2_dist_hidden_perturbed_from_original'].append(torch.norm(perturbed_hidden_states[t, s] - hidden_states[t]).item())
        for metric_name, values in pm.items():
            if values:
                features[f'{metric_name}_mean'] = float(np.mean(values))
                features[f'{metric_name}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
                features[f'{metric_name}_min'] = float(np.min(values))
                features[f'{metric_name}_max'] = float(np.max(values))
            else:
                for suf in ('mean', 'std', 'min', 'max'):
                    features[f'{metric_name}_{suf}'] = 0.0
        all_features.append(features)
    return all_features
MANIFEST = {'train': 'manifests/final/gqa_train_probe_train.jsonl', 'val': 'manifests/final/gqa_train_probe_val.jsonl'}

def _labels(model):
    lab = {}
    for l in open(DATA_DIR / 'pipeline' / model / 'judged.jsonl'):
        if l.strip():
            r = json.loads(l)
            lab[str(r['item_id'])] = int(r['correct'])
    return lab

def _items(model, split):
    if split in ('train', 'val'):
        ids = [json.loads(l)['item_id'] for l in open(MANIFEST[split]) if l.strip()]
        draws = {json.loads(l)['item_id']: json.loads(l) for l in open(DATA_DIR / 'draws' / 'gqa_train_probe.jsonl') if l.strip()}
        lab = _labels(model)
        elig = [iid for iid in ids if iid in draws and str(iid) in lab]
        out = []
        for iid in elig:
            out.append({'item_id': iid, 'setname': 'gqa_train_probe', 'dataset': 'gqa', 'label': lab[str(iid)], 'row': draws[iid], 'png': None, 'delta': None})
        return out
    meta = {}
    for f in glob.glob(str(CH / model / 'per_item_s*.jsonl')):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                meta[r['item_id']] = r
    out = []
    for iid, r in meta.items():
        out.append({'item_id': iid, 'setname': r['dataset'], 'dataset': DATASETS[r['dataset']], 'label': int(r['label']), 'row': None, 'question': r['question'], 'png': str(DATA_DIR / r['clean_png']), 'delta': str(DATA_DIR / r['delta_npy']), 'byte_identical': r['byte_identical']})
    return out

def run(model, split, image_mode='clean', gpu=0, shard=0, num_shards=8, redistribute=False, out_shard=None):
    assert image_mode in ('clean', 'adv')
    tag = split + ('_adv' if image_mode == 'adv' else '')
    d = OUT / model / tag
    d.mkdir(parents=True, exist_ok=True)
    items = _items(model, split)
    if image_mode == 'adv':
        items = [it for it in items if it.get('byte_identical')]
    n = len(items)
    if redistribute:
        done_all = set()
        for f in glob.glob(str(d / 'feat_s*.jsonl')):
            for l in open(f):
                if l.strip():
                    done_all.add(json.loads(l)['item_id'])
        items = [it for it in items if it['item_id'] not in done_all]
    mine = items[shard::num_shards]
    fidx = out_shard if out_shard is not None else shard
    outp = d / f'feat_s{fidx}.jsonl'
    done = set()
    if outp.exists():
        for l in open(outp):
            if l.strip():
                done.add(json.loads(l)['item_id'])
    todo = [it for it in mine if it['item_id'] not in done]
    print(f'[ccps-feat {model} {tag} s{shard}/{num_shards}] items={n} mine={len(mine)} done={len(done)} todo={len(todo)}', flush=True)
    if not todo:
        return
    dev = f'cuda:{gpu}'
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    adapter = build_adapter(model, device=dev)
    adapter.load()
    for p_ in adapter.model.parameters():
        p_.requires_grad_(False)
    lm_head = adapter.model.get_output_embeddings()
    lm_w = lm_head.weight
    ds_adapters = {ds: get_ds_adapter(ds) for ds in set(DATASETS.values())}
    fj = open(outp, 'a')
    nrec = 0
    for it in todo:
        iid = it['item_id']
        dataset = it['dataset']
        try:
            if it['png']:
                img = Image.open(it['png']).convert('RGB')
                question = it['question']
                spec = answer_spec(dataset, _draw_answer_type(it))
            else:
                row = it['row']
                di = DatasetItem.from_dict(row)
                spec = answer_spec(dataset, row.get('answer_type'))
                img = _ds(ds_adapters[dataset].get_image(di).convert('RGB'))
                question = di.question
            ans = adapter.generate_answer(img, question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
            if not ans.token_ids:
                continue
            inputs2, L = _score_inputs(adapter, img, question, ans.token_ids, spec)
            md = inputs2['pixel_values'].dtype
            if image_mode == 'adv':
                delta = torch.tensor(np.load(it['delta']).astype(np.float32), device=dev)
                inputs2 = dict(inputs2)
                inputs2['pixel_values'] = (inputs2['pixel_values'].float() + delta).to(md)
            with torch.no_grad():
                o = adapter.model(**inputs2, output_hidden_states=True, use_cache=False)
                hs = o.hidden_states[-1][0].float()
                lg = o.logits[0].float()
            T = len(ans.token_ids)
            pos = list(range(L - 1, L - 1 + T))
            H = hs[pos]
            LG = lg[pos]
            tid = torch.tensor(ans.token_ids, device=dev)
            jac = compute_jacobians_batch_analytical(H, LG, tid, lm_w.float())
            pert_h = generate_perturbation_trajectories_batch(H, jac, PEI_RADIUS, PEI_STEPS)
            pert_lg = get_perturbed_logits_batch(lm_head, pert_h.to(md)).float()
            feats = extract_ccps_features_batch(H, LG, jac, pert_h, pert_lg, tid, PEI_RADIUS, PEI_STEPS)
            for i, fd in enumerate(feats):
                rec = {'item_id': iid, 'hash_id': iid, 'is_correct': it['label'], 'token_idx': i, 'token_id': int(ans.token_ids[i]), 'dataset': it['setname']}
                rec.update({k: float(fd[k]) for k in FEATURE_COLUMNS})
                fj.write(json.dumps(rec) + '\n')
            fj.flush()
            nrec += 1
            if nrec % 100 == 0:
                torch.cuda.empty_cache()
                print(f'[ccps-feat {model} {tag} s{shard}] {nrec}/{len(todo)}', flush=True)
        except Exception as e:
            print(f'err {iid}: {str(e)[:140]}', flush=True)
    fj.close()
    print(json.dumps({'model': model, 'tag': tag, 'shard': shard, 'n_written': nrec}), flush=True)

def _process_item(adapter, lm_head, lm_w, ds_adapters, it, image_mode, dev):
    dataset = it['dataset']
    if it['png']:
        img = Image.open(it['png']).convert('RGB')
        question = it['question']
        spec = answer_spec(dataset, _draw_answer_type(it))
    else:
        row = it['row']
        di = DatasetItem.from_dict(row)
        spec = answer_spec(dataset, row.get('answer_type'))
        img = _ds(ds_adapters[dataset].get_image(di).convert('RGB'))
        question = di.question
    ans = adapter.generate_answer(img, question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
    if not ans.token_ids:
        return None
    inputs2, L = _score_inputs(adapter, img, question, ans.token_ids, spec)
    md = inputs2['pixel_values'].dtype
    if image_mode == 'adv':
        delta = torch.tensor(np.load(it['delta']).astype(np.float32), device=dev)
        inputs2 = dict(inputs2)
        inputs2['pixel_values'] = (inputs2['pixel_values'].float() + delta).to(md)
    with torch.no_grad():
        o = adapter.model(**inputs2, output_hidden_states=True, use_cache=False)
        hs = o.hidden_states[-1][0].float()
        lg = o.logits[0].float()
    T = len(ans.token_ids)
    pos = list(range(L - 1, L - 1 + T))
    H = hs[pos]
    LG = lg[pos]
    tid = torch.tensor(ans.token_ids, device=dev)
    jac = compute_jacobians_batch_analytical(H, LG, tid, lm_w.float())
    pert_h = generate_perturbation_trajectories_batch(H, jac, PEI_RADIUS, PEI_STEPS)
    pert_lg = get_perturbed_logits_batch(lm_head, pert_h.to(md)).float()
    feats = extract_ccps_features_batch(H, LG, jac, pert_h, pert_lg, tid, PEI_RADIUS, PEI_STEPS)
    rows = []
    for i, fd in enumerate(feats):
        rec = {'item_id': it['item_id'], 'hash_id': it['item_id'], 'is_correct': it['label'], 'token_idx': i, 'token_id': int(ans.token_ids[i]), 'dataset': it['setname']}
        rec.update({k: float(fd[k]) for k in FEATURE_COLUMNS})
        rows.append(rec)
    return rows
DRAIN_NSHARDS = 16
UNITS = [('test', 'clean'), ('test', 'adv'), ('val', 'clean'), ('train', 'clean')]

def drain(gpu, models, claimroot, units=None):
    import os
    torch.set_num_threads(1)
    units = units or UNITS
    dev = f'cuda:{gpu}'
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for model in models:
        claimdir = claimroot / model
        claimdir.mkdir(parents=True, exist_ok=True)
        keys = [f'{sp}_{mode}_{s}' for sp, mode in units for s in range(DRAIN_NSHARDS)]
        if all(((claimdir / k).exists() for k in keys)):
            continue
        print(f'[drain gpu{gpu}] loading {model}', flush=True)
        adapter = build_adapter(model, device=dev)
        adapter.load()
        for p_ in adapter.model.parameters():
            p_.requires_grad_(False)
        lm_head = adapter.model.get_output_embeddings()
        lm_w = lm_head.weight
        ds_adapters = {ds: get_ds_adapter(ds) for ds in set(DATASETS.values())}
        item_cache = {}
        for sp, mode in units:
            for s in range(DRAIN_NSHARDS):
                key = f'{sp}_{mode}_{s}'
                try:
                    os.mkdir(claimdir / key)
                except FileExistsError:
                    continue
                if (sp, mode) not in item_cache:
                    its = _items(model, sp)
                    if mode == 'adv':
                        its = [x for x in its if x.get('byte_identical')]
                    item_cache[sp, mode] = its
                allit = item_cache[sp, mode]
                mine = allit[s::DRAIN_NSHARDS]
                d = OUT / model / (sp + ('_adv' if mode == 'adv' else ''))
                d.mkdir(parents=True, exist_ok=True)
                outp = d / f'feat_s{s}.jsonl'
                done = set()
                if outp.exists():
                    for l in open(outp):
                        if l.strip():
                            done.add(json.loads(l)['item_id'])
                todo = [x for x in mine if x['item_id'] not in done]
                fj = open(outp, 'a')
                nrec = 0
                for it in todo:
                    try:
                        rows = _process_item(adapter, lm_head, lm_w, ds_adapters, it, mode, dev)
                        if rows is None:
                            continue
                        for r in rows:
                            fj.write(json.dumps(r) + '\n')
                        fj.flush()
                        nrec += 1
                        if nrec % 200 == 0:
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"err {it['item_id']}: {str(e)[:120]}", flush=True)
                fj.close()
                print(f'[drain gpu{gpu}] {model} {key}: wrote {nrec}', flush=True)
        del adapter
        torch.cuda.empty_cache()
    print(f'[drain gpu{gpu}] all models drained', flush=True)

def _draw_answer_type(it):
    d = {json.loads(l)['item_id']: json.loads(l).get('answer_type') for l in open(DATA_DIR / 'draws' / f"{it['setname']}.jsonl") if l.strip()}
    return d.get(it['item_id'])

def merge(model, tag):
    d = OUT / model / tag
    rows = []
    for f in glob.glob(str(d / 'feat_s*.jsonl')):
        for l in open(f):
            if l.strip():
                rows.append(json.loads(l))
    if not rows:
        print(f'no rows for {model}/{tag}')
        return None
    df = pd.DataFrame(rows)
    df.to_pickle(d / 'features.pkl')
    print(f"{model}/{tag}: {len(df)} token rows, {df['hash_id'].nunique()} samples -> features.pkl")
    return df
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--drain', action='store_true', help='work-stealing drain worker')
    ap.add_argument('--models', default='', help='comma-separated models for drain')
    ap.add_argument('--units', default='', help='comma-separated: train,val,test,test_adv (default all)')
    ap.add_argument('--claimroot', default=str(DATA_DIR / 'ccps' / 'claims'))
    ap.add_argument('--model', required=False)
    ap.add_argument('--split', choices=['train', 'val', 'test'])
    ap.add_argument('--image-mode', default='clean', choices=['clean', 'adv'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=8)
    ap.add_argument('--redistribute', action='store_true')
    ap.add_argument('--out-shard', type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    from pathlib import Path as _P
    if a.drain:
        units = None
        if a.units:
            um = {'train': ('train', 'clean'), 'val': ('val', 'clean'), 'test': ('test', 'clean'), 'test_adv': ('test', 'adv')}
            units = [um[u] for u in a.units.split(',') if u in um]
        drain(a.gpu, [m for m in a.models.split(',') if m], _P(a.claimroot), units)
    elif a.merge:
        merge(a.model, a.split + ('_adv' if a.image_mode == 'adv' else ''))
    else:
        run(a.model, a.split, a.image_mode, a.gpu, a.shard, a.num_shards, a.redistribute, a.out_shard)
