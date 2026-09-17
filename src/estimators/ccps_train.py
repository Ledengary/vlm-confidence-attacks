from __future__ import annotations
import argparse, copy, glob, json, pickle
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.config import DATA_DIR
from src.estimators.ccps_metrics import auroc, auprc, ece, brier
SEED = 23
FEAT = DATA_DIR / 'ccps' / 'features'
TRAINED = DATA_DIR / 'ccps' / 'trained'
PREDS = DATA_DIR / 'ccps' / 'preds'
HIDDEN_DIMS = [64, 32]
KERNEL_SIZES = [3, 3]
EMBED_DIM = 16
CLF_HIDDEN_DIMS = [32]
MAX_SEQ = 64
S1_LR = 0.0001
S1_WD = 0.1
S1_BS = 32
S1_STEPS = 5000
S1_MARGIN = 1.0
S2_LR = 0.0001
S2_WD = 0.0
S2_BS = 32
S2_MAX_EPOCHS = 200
S2_PATIENCE = 20
S2_AUROC_W = 0.6
EXCLUDE = {'hash_id', 'is_correct', 'token_idx', 'token_id', 'token_str', 'sample_id', 'token_text', 'dataset', 'split', 'item_id'}

class ConvEmbeddingNet(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.activation = nn.ReLU()
        self.conv_layers = nn.ModuleList([nn.Conv1d(input_dim, HIDDEN_DIMS[0], KERNEL_SIZES[0], padding=KERNEL_SIZES[0] // 2)])
        for i in range(1, len(HIDDEN_DIMS)):
            self.conv_layers.append(nn.Conv1d(HIDDEN_DIMS[i - 1], HIDDEN_DIMS[i], KERNEL_SIZES[i], padding=KERNEL_SIZES[i] // 2))
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(HIDDEN_DIMS[-1], EMBED_DIM)

    def forward(self, x, seq_lengths=None):
        x = x.transpose(1, 2)
        for c in self.conv_layers:
            x = self.activation(c(x))
        return self.fc(self.pool(x).squeeze(-1))

class ConvClassifierWithEmbedding(nn.Module):

    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = nn.Sequential(nn.Linear(EMBED_DIM, CLF_HIDDEN_DIMS[0]), nn.ReLU(), nn.Linear(CLF_HIDDEN_DIMS[0], 2))

    def forward(self, x, seq_lengths=None):
        emb = self.embedding_model(x, seq_lengths)
        return (self.classifier(emb), emb)

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        return torch.mean((1 - label) * dist.pow(2) + label * torch.clamp(self.margin - dist, min=0.0).pow(2))

class ContrastiveDS(Dataset):

    def __init__(self, samples, max_seq):
        self.s = samples
        self.m = max_seq

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        a = self.s[idx]
        o_idx = np.random.choice(len(self.s))
        while o_idx == idx:
            o_idx = np.random.choice(len(self.s))
        o = self.s[o_idx]
        d = a['features_scaled'].shape[1]
        ap = np.zeros((self.m, d))
        op = np.zeros((self.m, d))
        al = min(a['seq_length'], self.m)
        ol = min(o['seq_length'], self.m)
        ap[:al] = a['features_scaled'][:al]
        op[:ol] = o['features_scaled'][:ol]
        pl = 1 if a['label'] != o['label'] else 0
        return (torch.tensor(ap, dtype=torch.float32), torch.tensor(op, dtype=torch.float32), torch.tensor(pl, dtype=torch.float32), torch.tensor(al), torch.tensor(ol))

class ClfDS(Dataset):

    def __init__(self, samples, max_seq):
        self.s = samples
        self.m = max_seq

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        s = self.s[idx]
        d = s['features_scaled'].shape[1]
        sl = min(s['seq_length'], self.m)
        pad = np.zeros((self.m, d))
        pad[:sl] = s['features_scaled'][:sl]
        return (torch.tensor(pad, dtype=torch.float32), torch.tensor(s['label'], dtype=torch.long), torch.tensor(sl))

def _prepare(df, eps_high=20.0, feat_cols=None):
    feat_cols = feat_cols if feat_cols is not None else [c for c in df.columns if c not in EXCLUDE]
    out = []
    for hid in df['hash_id'].unique():
        sdf = df[df['hash_id'] == hid].sort_values('token_idx')
        feats = sdf[feat_cols].copy()
        if 'epsilon_to_flip_token' in feats.columns:
            feats['epsilon_to_flip_token'] = feats['epsilon_to_flip_token'].replace([np.inf, -np.inf], eps_high)
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
        out.append({'hash_id': hid, 'features': feats.values.astype(np.float64), 'seq_length': len(feats), 'label': int(sdf['is_correct'].iloc[0]), 'dataset': sdf['dataset'].iloc[0] if 'dataset' in sdf.columns else 'gqa'})
    return (out, feat_cols)

def _label_mapping(samples):
    labels = np.array([s['label'] for s in samples])
    swapped = bool(int((labels == 0).sum()) < int((labels == 1).sum()))
    lm = {'labels_swapped': swapped}
    if swapped:
        for s in samples:
            s['label'] = 1 - s['label']
    return (samples, lm)

def _apply_mapping(samples, lm):
    if lm.get('labels_swapped'):
        for s in samples:
            s['label'] = 1 - s['label']
    return samples

def all_metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {'auroc': auroc(y, p), 'auprc': auprc(y, p), 'ece': ece(y, p), 'brier': brier(y, p), 'accuracy': float(((p >= 0.5).astype(int) == y).mean())}

def _load_feats(model, tag):
    p = FEAT / model / tag / 'features.pkl'
    return pd.read_pickle(p)

def train(model, gpu=0, feat_subset=None, out_root=None):
    out_root = out_root or TRAINED
    dev = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tr, feat_cols = _prepare(_load_feats(model, 'train'), feat_cols=feat_subset)
    va, _ = _prepare(_load_feats(model, 'val'), feat_cols=feat_subset)
    tr, lm = _label_mapping(tr)
    va = _apply_mapping(va, lm)
    scaler = StandardScaler().fit(np.vstack([s['features'] for s in tr]))
    for s in tr:
        s['features_scaled'] = scaler.transform(s['features'])
    for s in va:
        s['features_scaled'] = scaler.transform(s['features'])
    fdim = tr[0]['features_scaled'].shape[1]
    print(f"[ccps-train {model}] train={len(tr)} val={len(va)} feat_dim={fdim} swapped={lm['labels_swapped']}", flush=True)
    enc = ConvEmbeddingNet(fdim).to(dev).train()
    crit = ContrastiveLoss(S1_MARGIN)
    opt = optim.AdamW(enc.parameters(), lr=S1_LR, weight_decay=S1_WD)
    loader = DataLoader(ContrastiveDS(tr, MAX_SEQ), batch_size=S1_BS, shuffle=True)
    step = 0
    while step < S1_STEPS:
        for a, o, l, al, ol in loader:
            if step >= S1_STEPS:
                break
            a, o, l = (a.to(dev), o.to(dev), l.to(dev))
            opt.zero_grad()
            loss = crit(enc(a), enc(o), l)
            loss.backward()
            opt.step()
            step += 1
    print(f'[ccps-train {model}] stage1 done ({S1_STEPS} steps)', flush=True)
    clf = ConvClassifierWithEmbedding(enc).to(dev)
    n_pos = sum((s['label'] == 1 for s in tr))
    n_neg = sum((s['label'] == 0 for s in tr))
    pw = n_neg / n_pos if n_pos > 0 else 1.0
    crit2 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], device=dev, dtype=torch.float))
    opt2 = optim.Adam(clf.parameters(), lr=S2_LR, weight_decay=S2_WD)
    trl = DataLoader(ClfDS(tr, MAX_SEQ), batch_size=S2_BS, shuffle=True)
    val = DataLoader(ClfDS(va, MAX_SEQ), batch_size=S2_BS, shuffle=False)

    def _eval():
        clf.eval()
        P, Y = ([], [])
        with torch.no_grad():
            for f, y, sl in val:
                lo, _ = clf(f.to(dev))
                P.append(F.softmax(lo, 1)[:, 1].cpu().numpy())
                Y.append(y.numpy())
        return all_metrics(np.concatenate(Y), np.concatenate(P))
    best_score, best_state, patience = (-1000000000.0, None, 0)
    for epoch in range(S2_MAX_EPOCHS):
        clf.train()
        for f, y, sl in trl:
            f, y = (f.to(dev), y.to(dev))
            opt2.zero_grad()
            lo, _ = clf(f)
            loss = crit2(lo, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt2.step()
        m = _eval()
        score = S2_AUROC_W * m['auroc'] + (1 - S2_AUROC_W) * (1 - m['ece'])
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(clf.state_dict())
            patience = 0
            best_m = m
            best_ep = epoch
        else:
            patience += 1
        if patience >= S2_PATIENCE:
            break
    if best_state is not None:
        clf.load_state_dict(best_state)
    print(f"[ccps-train {model}] stage2 best_ep={best_ep} val AUROC={best_m['auroc']:.4f} ECE={best_m['ece']:.4f} Brier={best_m['brier']:.4f}", flush=True)
    d = out_root / model / 'best'
    d.mkdir(parents=True, exist_ok=True)
    torch.save(clf.state_dict(), d / 'classifier_model.pt')
    pickle.dump(scaler, open(d / 'scaler.pkl', 'wb'))
    json.dump({'feature_dim': fdim, 'feature_cols': feat_cols, 'label_mapping': lm, 'max_seq_length': MAX_SEQ, 'hidden_dims': HIDDEN_DIMS, 'kernel_sizes': KERNEL_SIZES, 'embed_dim': EMBED_DIM, 'classifier_hidden_dims': CLF_HIDDEN_DIMS, 'val_metrics': best_m, 'best_epoch': best_ep, 'pos_weight_ratio': pw, 'seed': SEED}, open(d / 'model_info.json', 'w'), indent=2)
    print(f'[ccps-train {model}] saved -> {d}', flush=True)

def _load_model(model, dev, out_root=None):
    d = (out_root or TRAINED) / model / 'best'
    info = json.load(open(d / 'model_info.json'))
    enc = ConvEmbeddingNet(info['feature_dim'])
    clf = ConvClassifierWithEmbedding(enc)
    clf.load_state_dict(torch.load(d / 'classifier_model.pt', map_location='cpu'))
    clf.to(dev).eval()
    scaler = pickle.load(open(d / 'scaler.pkl', 'rb'))
    return (clf, scaler, info)

def predict(model, tag, gpu=0, out_root=None):
    dev = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    clf, scaler, info = _load_model(model, dev, out_root)
    lm = info['label_mapping']
    msl = info['max_seq_length']
    samples, _ = _prepare(_load_feats(model, tag), feat_cols=info.get('feature_cols'))
    for s in samples:
        s['features_scaled'] = scaler.transform(s['features'])
    conf = []
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            b = samples[i:i + 64]
            d = b[0]['features_scaled'].shape[1]
            X = np.zeros((len(b), msl, d))
            for j, s in enumerate(b):
                L = min(s['seq_length'], msl)
                X[j, :L] = s['features_scaled'][:L]
            lo, _ = clf(torch.tensor(X, dtype=torch.float32, device=dev))
            conf.extend(F.softmax(lo, 1)[:, 1].cpu().numpy().tolist())
    out = []
    for s, c in zip(samples, conf):
        pcorrect = 1 - c if lm.get('labels_swapped') else c
        out.append({'item_id': s['hash_id'], 'dataset': s['dataset'], 'label': int(s['label']), 'conf': float(pcorrect)})
    return (out, samples)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--mode', required=True, choices=['train', 'predict'])
    ap.add_argument('--tag', default='test')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    if a.mode == 'train':
        train(a.model, a.gpu)
    else:
        out, _ = predict(a.model, a.tag, a.gpu)
        d = PREDS / a.model
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f'{a.tag}.jsonl', 'w') as f:
            for r in out:
                f.write(json.dumps(r) + '\n')
        print(f'wrote {len(out)} preds -> {d}/{a.tag}.jsonl')
