from __future__ import annotations
import glob
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from src.config import DATA_DIR
from src.hidden_states.ccps_features import FEATURE_COLUMNS
from src.estimators.ccpsd import _ccps_conf_from_features, ccpsd_feature_tensor
from src.estimators.ccps_train import TRAINED as CCPS_TRAINED, _load_model as load_ccps_model
from src.estimators.train_probe import SAPLMANet
PIPE = DATA_DIR / 'pipeline'
COUPLED = DATA_DIR / 'coupled'
CCPSD_TRAINED = DATA_DIR / 'ccps' / 'trained_ccpsd'
ESTIMATORS = ['tokprob', 'pik', 'saplma', 'ccpsd', 'ccps', 'full', 'coupled']
ATTACKED = ['pik', 'saplma', 'ccpsd', 'full', 'coupled']
COUPLED_LOCUS = 'sap'
COUPLED_K = 8
HIDDEN_F16_ROUNDTRIP = True

def _f16_ste(h: torch.Tensor) -> torch.Tensor:
    if not HIDDEN_F16_ROUNDTRIP:
        return h
    hard = h.to(torch.float16).to(h.dtype)
    return h + (hard - h).detach()

def _basis_svd(diffs: np.ndarray) -> np.ndarray:
    U, Sv, Vt = np.linalg.svd(diffs, full_matrices=False)
    tol = max(diffs.shape) * np.finfo(np.float32).eps * (Sv[0] if Sv.size else 0.0)
    r = int((Sv > max(tol, 1e-06)).sum())
    return Vt[:r]

def load_coupled_meta(model: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(COUPLED / model / '*_meta_s*.jsonl'))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r['item_id']] = r
    return out

class EstimatorBank:

    def __init__(self, model: str, adapter, device: str):
        self.model = model
        self.adapter = adapter
        self.dev = device
        self.meta = {'model': model, 'sources': {}, 'orientation': {}}
        pm = json.load(open(PIPE / model / 'probes_meta.json'))
        self.probe = {}
        self.probe_swapped = {}
        for key in ('pik', 'saplma'):
            ch = pm['channels'][key]
            net = SAPLMANet(int(ch['input_dim']))
            state = torch.load(PIPE / model / 'probe_weights' / f'{key}.pth', map_location='cpu')
            net.load_state_dict(state)
            net.to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.probe[key] = net
            self.probe_swapped[key] = bool(ch['swapped'])
            self.meta['sources'][key] = str(PIPE / model / 'probe_weights' / f'{key}.pth')
            self.meta['orientation'][key] = 'P(correct) = 1 - sigmoid(logit), class 1 is incorrect' if ch['swapped'] else 'P(correct) = sigmoid(logit)'
        self.ccpsd_clf, sc_d, self.ccpsd_info = load_ccps_model(model, device, CCPSD_TRAINED)
        self.ccps_clf, sc_t, self.ccps_info = load_ccps_model(model, device, CCPS_TRAINED)
        for net in (self.ccpsd_clf, self.ccps_clf):
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
        self.ccpsd_mean = torch.tensor(sc_d.mean_, device=device, dtype=torch.float32)
        self.ccpsd_scale = torch.tensor(sc_d.scale_, device=device, dtype=torch.float32)
        self.ccps_mean = torch.tensor(sc_t.mean_, device=device, dtype=torch.float32)
        self.ccps_scale = torch.tensor(sc_t.scale_, device=device, dtype=torch.float32)
        self.ccpsd_swapped = bool(self.ccpsd_info['label_mapping'].get('labels_swapped'))
        self.ccps_swapped = bool(self.ccps_info['label_mapping'].get('labels_swapped'))
        self.ccpsd_msl = int(self.ccpsd_info['max_seq_length'])
        self.ccps_msl = int(self.ccps_info['max_seq_length'])
        self.ccps_cols = self.ccps_info.get('feature_cols', FEATURE_COLUMNS)
        self.meta['sources']['ccpsd'] = str(CCPSD_TRAINED / model / 'best')
        self.meta['sources']['ccps'] = str(CCPS_TRAINED / model / 'best')
        for k, sw in (('ccpsd', self.ccpsd_swapped), ('ccps', self.ccps_swapped)):
            self.meta['orientation'][k] = 'P(correct) = 1 - softmax(logits)[1], class 1 is incorrect' if sw else 'P(correct) = softmax(logits)[1]'
        self.coupled_net = {}
        for tag in ('FULL', 'COUPLED'):
            path = COUPLED / model / 'estimators' / f'{COUPLED_LOCUS}_k{COUPLED_K}_{tag}.pth'
            state = torch.load(path, map_location='cpu')
            dim = state['net.0.weight'].shape[1]
            net = SAPLMANet(dim)
            net.load_state_dict(state)
            net.to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.coupled_net[tag] = net
            self.meta['sources'][tag.lower()] = str(path)
            self.meta['orientation'][tag.lower()] = 'P(correct) = sigmoid(logit)'
        self.meta['orientation']['tokprob'] = 'already P(committed answer span), higher is confident'
        self.meta['sources']['tokprob'] = 'iclr2027/engine.score, committed channel definition'
        self.coupled_meta = load_coupled_meta(model)
        self._lm_head = adapter.model.get_output_embeddings()
        self._lm_w = self._lm_head.weight
        self._unembed_np = None
        self._basis_cache = {}

    def _unembed(self) -> np.ndarray:
        if self._unembed_np is None:
            self._unembed_np = self._lm_w.detach().to(torch.float32).cpu().numpy()
        return self._unembed_np

    def coupled_projector(self, item_id: str):
        if item_id in self._basis_cache:
            return self._basis_cache[item_id]
        rec = self.coupled_meta.get(item_id)
        if rec is None:
            self._basis_cache[item_id] = None
            return None
        W = self._unembed()
        wa = W[rec['ans_ids'][0]]
        wc = W[np.asarray(rec['comp_ids'][:COUPLED_K], dtype=np.int64)]
        B = _basis_svd(wc - wa[None, :])
        Bt = torch.tensor(B, device=self.dev, dtype=torch.float32)
        self._basis_cache[item_id] = Bt
        return Bt

    def score_all(self, ctx, out, item_id: str, want: tuple=tuple(ESTIMATORS)) -> dict:
        L, n = (ctx.L, len(ctx.ans_ids))
        hs = out.hidden_states[-1][0]
        res = {}
        if 'tokprob' in want:
            rows = out.logits[0, L - 1:L - 1 + n].float()
            logp = torch.log_softmax(rows, dim=-1)
            idx = torch.tensor(ctx.ans_ids, device=rows.device)
            res['tokprob'] = torch.exp(logp.gather(1, idx.view(-1, 1)).sum() / max(n, 1))
        for key, locus in (('pik', L - 1), ('saplma', hs.shape[0] - 1)):
            if key not in want:
                continue
            h = _f16_ste(hs[locus].float())
            logit = self.probe[key](h.unsqueeze(0))[0]
            p = torch.sigmoid(logit)
            res[key] = 1.0 - p if self.probe_swapped[key] else p
        if any((k in want for k in ('full', 'coupled'))):
            h_sap = hs[hs.shape[0] - 1].float()
            if 'full' in want:
                res['full'] = torch.sigmoid(self.coupled_net['FULL'](h_sap.unsqueeze(0))[0])
            if 'coupled' in want:
                B = self.coupled_projector(item_id)
                if B is None:
                    res['coupled'] = None
                else:
                    proj = B.t() @ (B @ h_sap)
                    res['coupled'] = torch.sigmoid(self.coupled_net['COUPLED'](proj.unsqueeze(0))[0])
        if 'ccpsd' in want or 'ccps' in want:
            pos = list(range(L - 1, L - 1 + n))
            H = hs[pos].float()
            LG = out.logits[0][pos].float()
            tid = torch.tensor(ctx.ans_ids, device=H.device)
            if 'ccpsd' in want:
                feats = ccpsd_feature_tensor(H, LG, tid, self._lm_head, self._lm_w.float(), ctx.model_dtype)
                res['ccpsd'] = _ccps_conf_from_features(self.ccpsd_clf, self.ccpsd_mean, self.ccpsd_scale, feats, self.ccpsd_msl, self.ccpsd_swapped, self.dev)
            if 'ccps' in want:
                with torch.no_grad():
                    feats75 = self.ccps_feature_tensor(H.detach(), LG.detach(), tid, ctx.model_dtype)
                    res['ccps'] = _ccps_conf_from_features(self.ccps_clf, self.ccps_mean, self.ccps_scale, feats75, self.ccps_msl, self.ccps_swapped, self.dev)
        return res

    def ccps_feature_tensor(self, H, LG, tid, md, pei_radius=20.0, pei_steps=5):
        T, D = H.shape
        V = LG.shape[-1]
        probs0 = F.softmax(LG, dim=-1)
        logp0 = F.log_softmax(LG, dim=-1)
        idx = torch.arange(T, device=H.device)
        lp_act = logp0[idx, tid]
        p_act = probs0[idx, tid]
        lg_act = LG[idx, tid]
        arg0 = LG.argmax(dim=-1)
        p_arg0 = probs0[idx, arg0]
        lg_arg0 = LG[idx, arg0]
        ent0 = -(probs0 * torch.log(probs0 + 1e-09)).sum(-1)
        top2_lg = LG.topk(2, dim=-1).values
        marg_lg0 = top2_lg[:, 0] - top2_lg[:, 1]
        top2_p = probs0.topk(2, dim=-1).values
        marg_p0 = top2_p[:, 0] - top2_p[:, 1]
        norm_lg0 = torch.norm(LG, dim=-1)
        std_lg0 = LG.std(dim=-1)
        norm_h0 = torch.norm(H, dim=-1)
        is_arg = (arg0 == tid).to(LG.dtype)
        y = F.one_hot(tid, V).to(LG.dtype)
        jac = torch.matmul(probs0 - y, self._lm_w.float())
        jnorm = torch.norm(jac, dim=-1)
        direction = jac / (jnorm.unsqueeze(-1) + 1e-12)
        delta_r = pei_radius / pei_steps
        radii = torch.arange(1, pei_steps + 1, device=H.device, dtype=H.dtype) * delta_r
        pert_h = H.unsqueeze(1) + radii.view(1, -1, 1) * direction.unsqueeze(1)
        pert_lg = self._lm_head(pert_h.reshape(-1, D).to(md)).reshape(T, pei_steps, -1).float()
        pprobs = F.softmax(pert_lg, dim=-1)
        plogp = F.log_softmax(pert_lg, dim=-1)
        si = torch.arange(pei_steps, device=H.device).view(1, -1)
        pi = idx.view(T, 1)
        plp_act = plogp[pi, si, tid.view(T, 1)]
        pp_act = pprobs[pi, si, tid.view(T, 1)]
        plg_act = pert_lg[pi, si, tid.view(T, 1)]
        dlp = lp_act.unsqueeze(1) - plp_act
        parg = pert_lg.argmax(dim=-1)
        pp_arg = pprobs.gather(2, parg.unsqueeze(-1)).squeeze(-1)
        plg_arg = pert_lg.gather(2, parg.unsqueeze(-1)).squeeze(-1)
        changed = (parg != tid.view(T, 1)).to(LG.dtype)
        pent = -(pprobs * torch.log(pprobs + 1e-09)).sum(-1)
        ptop2 = pert_lg.topk(2, dim=-1).values
        pmarg = ptop2[..., 0] - ptop2[..., 1]
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
        flipped = parg != tid.view(T, 1)
        eps_flip = torch.full((T,), 20.0, device=H.device, dtype=LG.dtype)
        for s in range(pei_steps - 1, -1, -1):
            eps_flip = torch.where(flipped[:, s], radii[s].expand(T), eps_flip)

        def ms(x):
            return (x.mean(-1), x.std(-1, unbiased=False), x.min(-1).values, x.max(-1).values)
        col = {'original_log_prob_actual': lp_act, 'original_prob_actual': p_act, 'original_logit_actual': lg_act, 'original_prob_argmax': p_arg0, 'original_logit_argmax': lg_arg0, 'original_entropy': ent0, 'original_margin_logit_top1_top2': marg_lg0, 'original_margin_prob_top1_top2': marg_p0, 'original_norm_logits_L2': norm_lg0, 'original_std_logits': std_lg0, 'original_norm_hidden_state_L2': norm_h0, 'is_actual_token_original_argmax': is_arg, 'jacobian_norm_token': jnorm, 'epsilon_to_flip_token': eps_flip, 'pei_value_token': pei}
        for name, x in (('perturbed_log_prob_actual', plp_act), ('perturbed_prob_actual', pp_act), ('perturbed_logit_actual', plg_act), ('delta_log_prob_actual_from_original', dlp), ('perturbed_prob_argmax', pp_arg), ('perturbed_logit_argmax', plg_arg), ('did_argmax_change_from_original', changed), ('perturbed_entropy', pent), ('perturbed_margin_logit_top1_top2', pmarg), ('perturbed_norm_logits_L2', pnorm_lg), ('kl_div_perturbed_from_original', kl), ('js_div_perturbed_from_original', js), ('cosine_sim_logits_perturbed_to_original', cos_lg), ('cosine_sim_hidden_perturbed_to_original', cos_h), ('l2_dist_hidden_perturbed_from_original', l2_h)):
            mu, sd, mn, mx = ms(x)
            col[f'{name}_mean'] = mu
            col[f'{name}_std'] = sd
            col[f'{name}_min'] = mn
            col[f'{name}_max'] = mx
        feats = torch.stack([col[c] for c in self.ccps_cols], dim=1)
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
PIK_SAP_SOURCE = 'attack_line'

def committed_clean(model: str, source: str=PIK_SAP_SOURCE) -> dict:
    out = {k: {} for k in ESTIMATORS}
    if source == 'attack_line':
        for f in sorted(glob.glob(str(COUPLED / model / '*_meta_s*.jsonl'))):
            for line in open(f):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    c = r.get('clean') or {}
                    for key in ('pik', 'saplma'):
                        if key in c:
                            out[key][r['item_id']] = float(c[key])
    else:
        for key in ('pik', 'saplma'):
            for f in sorted(glob.glob(str(PIPE / model / 'probe_conf' / f'{key}__*.json'))):
                out[key].update({k: float(v) for k, v in json.load(open(f)).items()})
    for key, root in (('ccps', DATA_DIR / 'ccps' / 'preds'), ('ccpsd', DATA_DIR / 'ccps' / 'preds_ccpsd')):
        f = root / model / 'test.jsonl'
        if f.exists():
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    out[key][r['item_id']] = float(r['conf'])
    import pandas as pd
    p = COUPLED / model / 'estimator_per_item.parquet'
    if p.exists():
        df = pd.read_parquet(p)
        df = df[(df['locus'] == COUPLED_LOCUS) & (df['k'] == COUPLED_K)]
        for tag, key in (('FULL', 'full'), ('COUPLED', 'coupled')):
            sub = df[df['estimator'] == tag]
            out[key] = dict(zip(sub['item_id'].tolist(), sub['p_clean'].astype(float).tolist()))
    return out
