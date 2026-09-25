from __future__ import annotations
import torch
import torch.nn.functional as F
from src.hidden_states.ccps_features import FEATURE_COLUMNS, PEI_RADIUS, PEI_STEPS
DIFF_FEATURES = ['original_log_prob_actual', 'original_prob_actual', 'original_logit_actual', 'original_entropy', 'original_norm_logits_L2', 'original_std_logits', 'original_norm_hidden_state_L2', 'jacobian_norm_token', 'pei_value_token', 'perturbed_log_prob_actual_mean', 'perturbed_log_prob_actual_std', 'perturbed_prob_actual_mean', 'perturbed_prob_actual_std', 'perturbed_logit_actual_mean', 'perturbed_logit_actual_std', 'delta_log_prob_actual_from_original_mean', 'delta_log_prob_actual_from_original_std', 'perturbed_entropy_mean', 'perturbed_entropy_std', 'perturbed_norm_logits_L2_mean', 'perturbed_norm_logits_L2_std', 'kl_div_perturbed_from_original_mean', 'kl_div_perturbed_from_original_std', 'js_div_perturbed_from_original_mean', 'js_div_perturbed_from_original_std', 'cosine_sim_logits_perturbed_to_original_mean', 'cosine_sim_logits_perturbed_to_original_std', 'cosine_sim_hidden_perturbed_to_original_mean', 'cosine_sim_hidden_perturbed_to_original_std', 'l2_dist_hidden_perturbed_from_original_mean', 'l2_dist_hidden_perturbed_from_original_std']
assert all((f in FEATURE_COLUMNS for f in DIFF_FEATURES)) and len(DIFF_FEATURES) == 31

def ccpsd_feature_tensor(H, LG, tid, lm_head, lm_w, md, pei_radius=PEI_RADIUS, pei_steps=PEI_STEPS):
    T, D = H.shape
    probs0 = F.softmax(LG, dim=-1)
    logp0 = F.log_softmax(LG, dim=-1)
    idx = torch.arange(T, device=H.device)
    lp_act = logp0[idx, tid]
    p_act = probs0[idx, tid]
    lg_act = LG[idx, tid]
    ent0 = -(probs0 * torch.log(probs0 + 1e-09)).sum(-1)
    norm_lg0 = torch.norm(LG, dim=-1)
    std_lg0 = LG.std(dim=-1)
    norm_h0 = torch.norm(H, dim=-1)
    y = F.one_hot(tid, LG.shape[-1]).to(LG.dtype)
    jac = torch.matmul(probs0 - y, lm_w)
    jnorm = torch.norm(jac, dim=-1)
    direction = jac / (jnorm.unsqueeze(-1) + 1e-12)
    delta_r = pei_radius / pei_steps
    radii = torch.arange(1, pei_steps + 1, device=H.device, dtype=H.dtype) * delta_r
    pert_h = H.unsqueeze(1) + radii.view(1, -1, 1) * direction.unsqueeze(1)
    pert_lg = lm_head(pert_h.reshape(-1, D).to(md)).reshape(T, pei_steps, -1).float()
    pprobs = F.softmax(pert_lg, dim=-1)
    plogp = F.log_softmax(pert_lg, dim=-1)
    pi = idx.view(T, 1)
    plp_act = plogp[pi, torch.arange(pei_steps, device=H.device).view(1, -1), tid.view(T, 1)]
    pp_act = pprobs[pi, torch.arange(pei_steps, device=H.device).view(1, -1), tid.view(T, 1)]
    plg_act = pert_lg[pi, torch.arange(pei_steps, device=H.device).view(1, -1), tid.view(T, 1)]
    dlp = lp_act.unsqueeze(1) - plp_act
    pent = -(pprobs * torch.log(pprobs + 1e-09)).sum(-1)
    pnorm_lg = torch.norm(pert_lg, dim=-1)
    p0 = probs0.unsqueeze(1)
    kl = (p0 * (torch.log(p0 + 1e-12) - plogp)).sum(-1)
    m = 0.5 * (p0 + pprobs)
    logm = torch.log(m + 1e-12)
    js = 0.5 * (m * (logm - logp0.unsqueeze(1))).sum(-1) + 0.5 * (m * (logm - plogp)).sum(-1)
    cos_lg = F.cosine_similarity(LG.unsqueeze(1).expand_as(pert_lg), pert_lg, dim=-1)
    cos_h = F.cosine_similarity(H.unsqueeze(1).expand_as(pert_h), pert_h, dim=-1)
    l2_h = torch.norm(pert_h - H.unsqueeze(1), dim=-1)
    fvals = F.relu(lp_act.unsqueeze(1) - plp_act)
    fpad = torch.cat([torch.zeros(T, 1, device=H.device, dtype=fvals.dtype), fvals], dim=1)
    pei = ((fpad[:, :-1] + fpad[:, 1:]) * 0.5).sum(-1) / pei_steps

    def ms(x):
        return (x.mean(-1), x.std(-1, unbiased=False))
    feats = [lp_act, p_act, lg_act, ent0, norm_lg0, std_lg0, norm_h0, jnorm, pei]
    for x in (plp_act, pp_act, plg_act, dlp, pent, pnorm_lg, kl, js, cos_lg, cos_h, l2_h):
        mu, sd = ms(x)
        feats += [mu, sd]
    return torch.stack(feats, dim=1)

def _ccps_conf_from_features(clf, scaler_mean, scaler_scale, feats, msl, swapped, dev):
    scaled = (feats - scaler_mean) / scaler_scale
    T, n = scaled.shape
    x = torch.zeros(1, msl, n, device=dev, dtype=scaled.dtype)
    L = min(T, msl)
    x[0, :L] = scaled[:L]
    logits, _ = clf(x)
    p1 = F.softmax(logits, dim=1)[0, 1]
    return 1 - p1 if swapped else p1

def attack_ccpsd(model, gpu=0, shard=0, num_shards=8, redistribute=False, out_shard=None, steps=15, eps_over_255=8, tau_safe=1.0):
    import glob, json, numpy as np
    from PIL import Image
    from src.config import DATA_DIR
    from src.estimators.channels import answer_spec
    from src.hidden_states.common import _ds
    from src.estimators.coupled_readout import _score_inputs, image_std_mean
    from src.estimators.coupled_readout import _greedy, _answer_preserved
    from src.hidden_states.ccps_features import compute_jacobians_batch_analytical, generate_perturbation_trajectories_batch, get_perturbed_logits_batch, extract_ccps_features_batch, FEATURE_COLUMNS, _draw_answer_type, DATASETS
    from src.estimators.ccps_train import _load_model, TRAINED
    import torch as T_
    dev = f'cuda:{gpu}'
    T_.manual_seed(23)
    np.random.seed(23)
    CCPSD_ROOT = DATA_DIR / 'ccps' / 'trained_ccpsd'
    clfd, scd, infod = _load_model(model, dev, CCPSD_ROOT)
    clft, sct, infot = _load_model(model, dev, TRAINED)
    md = clfd.to(dev).eval()
    clft.to(dev).eval()
    for p in clfd.parameters():
        p.requires_grad_(False)
    for p in clft.parameters():
        p.requires_grad_(False)
    scd_m = T_.tensor(scd.mean_, device=dev, dtype=T_.float32)
    scd_s = T_.tensor(scd.scale_, device=dev, dtype=T_.float32)
    sct_m = T_.tensor(sct.mean_, device=dev, dtype=T_.float32)
    sct_s = T_.tensor(sct.scale_, device=dev, dtype=T_.float32)
    sw_d = infod['label_mapping'].get('labels_swapped')
    sw_t = infot['label_mapping'].get('labels_swapped')
    msl_d = infod['max_seq_length']
    msl_t = infot['max_seq_length']
    tcols_t = infot.get('feature_cols', FEATURE_COLUMNS)
    from src.inference.models.factory import build_adapter
    from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
    ad = build_adapter(model, device=dev)
    ad.load()
    for p in ad.model.parameters():
        p.requires_grad_(False)
    lm_head = ad.model.get_output_embeddings()
    lm_w = lm_head.weight
    std = image_std_mean(ad)
    eps_pv = eps_over_255 / 255.0 / std
    alpha = eps_pv / 4
    ds_adapters = {ds: get_ds_adapter(ds) for ds in set(DATASETS.values())}
    CH = DATA_DIR / 'adv' / 'channels'
    meta = {}
    for f in glob.glob(str(CH / model / 'per_item_s*.jsonl')):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                meta[r['item_id']] = r
    items = [r for r in meta.values() if r.get('byte_identical')]
    items.sort(key=lambda r: r['item_id'])
    d = DATA_DIR / 'ccps' / 'ccpsd_attack' / model
    d.mkdir(parents=True, exist_ok=True)
    if redistribute:
        done_all = set()
        for f in glob.glob(str(d / 's*.jsonl')):
            for l in open(f):
                if l.strip():
                    done_all.add(json.loads(l)['item_id'])
        items = [r for r in items if r['item_id'] not in done_all]
    mine = items[shard::num_shards]
    fidx = out_shard if out_shard is not None else shard
    outp = d / f's{fidx}.jsonl'
    done = set()
    if outp.exists():
        for l in open(outp):
            if l.strip():
                done.add(json.loads(l)['item_id'])
    todo = [r for r in mine if r['item_id'] not in done]
    print(f'[ccpsd-attack {model} s{shard}] items={len(items)} mine={len(mine)} todo={len(todo)}', flush=True)
    fj = open(outp, 'a')
    nrec = 0
    for r in todo:
        iid = r['item_id']
        setname = r['dataset']
        dataset = DATASETS[setname]
        group = r['group']
        sgn = -1.0 if group == 'correct' else 1.0
        try:
            img = Image.open(DATA_DIR / r['clean_png']).convert('RGB')
            question = r['question']
            spec = answer_spec(dataset, _draw_answer_type({'item_id': iid, 'setname': setname}))
            ans = ad.generate_answer(img, question, format_instruction=spec.format_instruction, max_new_tokens=spec.max_new_tokens)
            if not ans.token_ids:
                continue
            inputs2, L = _score_inputs(ad, img, question, ans.token_ids, spec)
            mdt = inputs2['pixel_values'].dtype
            pv0 = inputs2['pixel_values'].detach().float()
            tid = T_.tensor(ans.token_ids, device=dev)
            pos = list(range(L - 1, L - 1 + len(ans.token_ids)))

            def ccpsd_conf(pv):
                inp = dict(inputs2)
                inp['pixel_values'] = pv.to(mdt)
                o = ad.model(**inp, output_hidden_states=True, use_cache=False)
                H = o.hidden_states[-1][0][pos].float()
                LG = o.logits[0][pos].float()
                feats = ccpsd_feature_tensor(H, LG, tid, lm_head, lm_w.float(), mdt)
                return (_ccps_conf_from_features(clfd, scd_m, scd_s, feats, msl_d, sw_d, dev), o)
            with T_.no_grad():
                cd_clean, _ = ccpsd_conf(pv0)
                cd_clean = float(cd_clean)
            delta = T_.zeros_like(pv0)
            best_delta = T_.zeros_like(pv0)
            best_mv = 0.0
            found = False
            for _ in range(steps):
                delta.requires_grad_(True)
                conf, o = ccpsd_conf(pv0 + delta)
                loss = conf if group == 'correct' else -conf
                grad = T_.autograd.grad(loss, delta)[0]
                with T_.no_grad():
                    preserved = _answer_preserved(o.logits[0].float(), L, ans.token_ids, tau_safe)
                    mv = cd_clean - float(conf) if group == 'correct' else float(conf) - cd_clean
                    if preserved and mv > best_mv:
                        best_mv = mv
                        best_delta = delta.detach().clone()
                        found = True
                    delta = (delta - alpha * grad.sign()).clamp(-eps_pv, eps_pv).detach()
            adv_pv = pv0 + best_delta
            with T_.no_grad():
                cd_adv, _ = ccpsd_conf(adv_pv)
                cd_adv = float(cd_adv)
                adv_answer = _greedy(ad, img, question, spec, adv_pv, mdt)
                bi = adv_answer == ans.answer_text

                def true_ccps_conf(pv):
                    inp = dict(inputs2)
                    inp['pixel_values'] = pv.to(mdt)
                    o2 = ad.model(**inp, output_hidden_states=True, use_cache=False)
                    H = o2.hidden_states[-1][0][pos].float()
                    LG = o2.logits[0][pos].float()
                    jac = compute_jacobians_batch_analytical(H, LG, tid, lm_w.float())
                    ph = generate_perturbation_trajectories_batch(H, jac, 20.0, 5)
                    pl = get_perturbed_logits_batch(lm_head, ph.to(mdt)).float()
                    fl = extract_ccps_features_batch(H, LG, jac, ph, pl, tid, 20.0, 5)
                    ft = T_.tensor([[fd[c] for c in tcols_t] for fd in fl], device=dev, dtype=T_.float32)
                    ft = T_.nan_to_num(ft, posinf=20.0, neginf=20.0)
                    return float(_ccps_conf_from_features(clft, sct_m, sct_s, ft, msl_t, sw_t, dev))
                tc_clean = true_ccps_conf(pv0)
                tc_adv = true_ccps_conf(adv_pv)
            fj.write(json.dumps({'item_id': iid, 'dataset': setname, 'group': group, 'label': int(r['label']), 'byte_identical': bi, 'found': found, 'ccpsd_clean': round(cd_clean, 5), 'ccpsd_adv': round(cd_adv, 5), 'trueccps_clean': round(tc_clean, 5), 'trueccps_adv': round(tc_adv, 5)}) + '\n')
            fj.flush()
            nrec += 1
            if nrec % 25 == 0:
                T_.cuda.empty_cache()
                print(f'[ccpsd-attack {model} s{shard}] {nrec}/{len(todo)}', flush=True)
        except Exception as e:
            print(f'err {iid}: {str(e)[:140]}', flush=True)
    fj.close()
    print(json.dumps({'model': model, 'shard': shard, 'n_written': nrec}), flush=True)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['attack'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=8)
    ap.add_argument('--redistribute', action='store_true')
    ap.add_argument('--out-shard', type=int, default=None)
    a = ap.parse_args()
    attack_ccpsd(a.model, a.gpu, a.shard, a.num_shards, a.redistribute, a.out_shard)
