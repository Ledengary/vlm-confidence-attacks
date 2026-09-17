from __future__ import annotations
import hashlib
import json
import pickle
import torch
import torch.nn.functional as F
from src.config import DATA_DIR
from src.hidden_states.ccps_features import FEATURE_COLUMNS
from src.estimators.ccps_train import TRAINED as CCPS_FROZEN, ConvClassifierWithEmbedding, ConvEmbeddingNet
from src.estimators.ccpsd import DIFF_FEATURES
from src.estimators.train_probe import SAPLMANet
from src.estimators.readout_bank import ESTIMATORS, EstimatorBank
V11 = DATA_DIR / 'snowglobe' / 'pfmc_v1_1'
V12 = DATA_DIR / 'snowglobe' / 'pfmc_v1_2'
CCPS_REPAIRED = V11 / 'ccps_repaired'
PROBES_V12 = V12 / 'internvl_probes'
CCPSD_FROZEN = DATA_DIR / 'ccps' / 'trained_ccpsd'
ECDF_V12 = V12 / 'ecdf_canonical.json'

def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def _load_conv(root, model):
    d = root / model / 'best'
    info = json.load(open(d / 'model_info.json'))
    enc = ConvEmbeddingNet(info['feature_dim'])
    clf = ConvClassifierWithEmbedding(enc)
    clf.load_state_dict(torch.load(d / 'classifier_model.pt', map_location='cpu'))
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    sc = pickle.load(open(d / 'scaler.pkl', 'rb'))
    return (clf, sc, info, d)

def component_hashes(model: str) -> dict:
    out = {}
    for tag, root in (('ccps', CCPS_REPAIRED / 'ccps'), ('ccpsd', CCPS_REPAIRED / 'ccpsd')):
        d = root / model / 'best'
        info = json.load(open(d / 'model_info.json'))
        out[tag] = {'classifier_weights': sha256_file(d / 'classifier_model.pt'), 'scaler': sha256_file(d / 'scaler.pkl'), 'model_info': sha256_file(d / 'model_info.json'), 'feature_column_manifest': hashlib.sha256(json.dumps(info['feature_cols']).encode()).hexdigest(), 'n_features': info['feature_dim'], 'dropped_columns': info.get('dropped_columns', []), 'path': str(d)}
    for key in ('pik', 'saplma'):
        p = PROBES_V12 / f'{key}.pth'
        m = PROBES_V12 / f'{key}_meta.json'
        if p.exists() and m.exists():
            out[key] = {'weights': sha256_file(p), 'meta': sha256_file(m), 'path': str(p), 'line': 'uncapped matched retrain'}
    if ECDF_V12.exists():
        out['ecdf_reference'] = {'path': str(ECDF_V12), 'sha256': sha256_file(ECDF_V12)}
    return out

class CanonicalBank(EstimatorBank):

    def __init__(self, model: str, adapter, device: str, diagnostic_legacy: bool=False, use_v12_probes: bool=True):
        super().__init__(model, adapter, device)
        self.model_key = model
        self.canonical = {'ccps': 'legacy', 'ccpsd': 'legacy', 'pik': 'frozen', 'saplma': 'frozen'}
        self.legacy = {}
        if diagnostic_legacy:
            self.legacy = {'ccps': _load_conv(CCPS_FROZEN, model), 'ccpsd': _load_conv(CCPSD_FROZEN, model)}
        for tag, root in (('ccps', CCPS_REPAIRED / 'ccps'), ('ccpsd', CCPS_REPAIRED / 'ccpsd')):
            d = root / model / 'best'
            if not (d / 'classifier_model.pt').exists():
                continue
            clf, sc, info, _ = _load_conv(root, model)
            base = FEATURE_COLUMNS if tag == 'ccps' else DIFF_FEATURES
            cols = info['feature_cols']
            keep = set(cols)
            idx = [i for i, c in enumerate(base) if c in keep]
            if len(idx) != len(cols):
                raise ValueError(f'{tag}/{model}: repaired columns are not a subset of the base set in order ({len(idx)} vs {len(cols)})')
            setattr(self, f'{tag}_clf', clf.to(device))
            setattr(self, f'{tag}_mean', torch.tensor(sc.mean_, device=device, dtype=torch.float32))
            setattr(self, f'{tag}_scale', torch.tensor(sc.scale_, device=device, dtype=torch.float32))
            setattr(self, f'{tag}_info', info)
            setattr(self, f'{tag}_cols', cols)
            setattr(self, f'{tag}_keep_idx', idx)
            setattr(self, f'{tag}_swapped', bool(info['label_mapping'].get('labels_swapped')))
            setattr(self, f'{tag}_msl', int(info['max_seq_length']))
            self.canonical[tag] = f"repaired_{info['feature_dim']}col"
            self.meta['sources'][tag] = str(d)
        if use_v12_probes and model == 'internvl3_5_2b':
            for key in ('pik', 'saplma'):
                p = PROBES_V12 / f'{key}.pth'
                m = PROBES_V12 / f'{key}_meta.json'
                if p.exists() and m.exists():
                    meta = json.load(open(m))
                    net = SAPLMANet(int(meta['input_dim']))
                    net.load_state_dict(torch.load(p, map_location='cpu'))
                    net.to(device).eval()
                    for q in net.parameters():
                        q.requires_grad_(False)
                    self.probe[key] = net
                    self.probe_swapped[key] = bool(meta['swapped'])
                    self.canonical[key] = 'uncapped_matched_retrain'
                    self.meta['sources'][key] = str(p)

    def _feats(self, tag: str, ctx, out):
        from src.estimators.ccpsd import ccpsd_feature_tensor
        L, n = (ctx.L, len(ctx.ans_ids))
        pos = list(range(L - 1, L - 1 + n))
        H = out.hidden_states[-1][0][pos].float()
        LG = out.logits[0][pos].float()
        tid = torch.tensor(ctx.ans_ids, device=H.device)
        if tag == 'ccpsd':
            feats = ccpsd_feature_tensor(H, LG, tid, self._lm_head, self._lm_w.float(), ctx.model_dtype)
        else:
            feats = self.ccps_feature_tensor(H, LG, tid, ctx.model_dtype)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        idx = getattr(self, f'{tag}_keep_idx', None)
        return feats if idx is None else feats[:, idx]

    def logit_and_score(self, tag: str, ctx, out):
        feats = self._feats(tag, ctx, out)
        mean = getattr(self, f'{tag}_mean')
        scale = getattr(self, f'{tag}_scale')
        msl = getattr(self, f'{tag}_msl')
        clf = getattr(self, f'{tag}_clf')
        swapped = getattr(self, f'{tag}_swapped')
        scaled = (feats - mean) / scale
        T, nf = scaled.shape
        if nf != len(getattr(self, f'{tag}_cols')):
            raise ValueError(f'{tag}: feature width {nf} does not match the repaired column set')
        x = torch.zeros(1, msl, nf, device=scaled.device, dtype=scaled.dtype)
        Ln = min(T, msl)
        x[0, :Ln] = scaled[:Ln]
        logits, _ = clf(x)
        d = logits[0, 1] - logits[0, 0]
        return (-d, torch.sigmoid(-d)) if swapped else (d, torch.sigmoid(d))

    def legacy_score(self, tag: str, ctx, out):
        if tag not in self.legacy:
            return None
        clf, sc, info, _ = self.legacy[tag]
        from src.estimators.ccpsd import ccpsd_feature_tensor
        L, n = (ctx.L, len(ctx.ans_ids))
        pos = list(range(L - 1, L - 1 + n))
        H = out.hidden_states[-1][0][pos].float()
        LG = out.logits[0][pos].float()
        tid = torch.tensor(ctx.ans_ids, device=H.device)
        if tag == 'ccpsd':
            feats = ccpsd_feature_tensor(H, LG, tid, self._lm_head, self._lm_w.float(), ctx.model_dtype)
        else:
            saved = self.ccps_cols
            try:
                self.ccps_cols = info.get('feature_cols', FEATURE_COLUMNS)
                feats = self.ccps_feature_tensor(H, LG, tid, ctx.model_dtype)
            finally:
                self.ccps_cols = saved
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        mean = torch.tensor(sc.mean_, device=feats.device, dtype=torch.float32)
        scale = torch.tensor(sc.scale_, device=feats.device, dtype=torch.float32)
        scaled = (feats - mean) / scale
        msl = int(info['max_seq_length'])
        x = torch.zeros(1, msl, scaled.shape[1], device=scaled.device, dtype=scaled.dtype)
        Ln = min(scaled.shape[0], msl)
        x[0, :Ln] = scaled[:Ln]
        with torch.no_grad():
            logits, _ = clf.to(feats.device)(x)
            p1 = F.softmax(logits, 1)[0, 1]
        return float(1 - p1) if info['label_mapping'].get('labels_swapped') else float(p1)

    def score_all(self, ctx, out, item_id: str, want=tuple(ESTIMATORS)) -> dict:
        rest = tuple((w for w in want if w not in ('ccps', 'ccpsd')))
        res = super().score_all(ctx, out, item_id, want=rest) if rest else {}
        for tag in ('ccps', 'ccpsd'):
            if tag in want:
                _, s = self.logit_and_score(tag, ctx, out)
                res[tag] = s
        return res
