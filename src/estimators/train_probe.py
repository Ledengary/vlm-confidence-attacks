from __future__ import annotations
import argparse
import copy
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from src.config import DATA_DIR, SEED
from src.estimators.metrics_lite import roc_auc_score, average_precision_score, brier_score_loss
HID = DATA_DIR / 'saplma' / 'hidden'
OUT = DATA_DIR / 'saplma' / 'probe_conf'
WEIGHTS = DATA_DIR / 'saplma' / 'probe_weights'
from src.config import MODEL_REGISTRY
MODELS = list(MODEL_REGISTRY)
EVAL_SETS = ['gqa_val_eval', 'vqav2_ood', 'pope_ood']
ARCH = (256, 128, 64)
MAX_EPOCHS, PATIENCE, BATCH, LR = (200, 20, 32, 0.001)

class SAPLMANet(nn.Module):

    def __init__(self, d):
        super().__init__()
        dims = [d] + list(ARCH)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_set(model, setname):
    hid, ids, labels = ([], [], [])
    for f in sorted(glob.glob(str(HID / model / setname / 'shard_*.npz'))):
        d = np.load(f, allow_pickle=True)
        hid.append(d['hidden'].astype(np.float32))
        ids.append(d['ids'])
        labels.append(d['labels'])
    if not hid:
        return (None, None, None)
    return (np.concatenate(hid), np.concatenate(ids), np.concatenate(labels).astype(int))

def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def _metrics(p_correct, y_correct):
    p = np.asarray(p_correct, float)
    y = np.asarray(y_correct, float)
    if len(set(y.tolist())) < 2:
        return {'n': int(len(y)), 'auroc': None, 'auprc': None, 'ece': round(_ece(p, y), 4), 'brier': round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4), 'base_rate': round(float(y.mean()), 4)}
    return {'n': int(len(y)), 'auroc': round(float(roc_auc_score(y, p)), 4), 'auprc': round(float(average_precision_score(y, p)), 4), 'ece': round(_ece(p, y), 4), 'brier': round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4), 'base_rate': round(float(y.mean()), 4)}

def label_mapping(y_true):
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n0 < n1:
        return (True, {'labels_swapped': True, 'class_1_represents': 'incorrect', 'n_correct': n1, 'n_incorrect': n0})
    return (False, {'labels_swapped': False, 'class_1_represents': 'correct', 'n_correct': n1, 'n_incorrect': n0})

def train_one(model, device='cuda:0'):
    Xtr, _, ytr_true = load_set(model, 'gqa_train_probe_train')
    Xva, _, yva_true = load_set(model, 'gqa_train_probe_val')
    if Xtr is None or Xva is None:
        print(f'{model}: hidden states missing')
        return None
    swapped, mapping = label_mapping(ytr_true)
    ytr = 1 - ytr_true if swapped else ytr_true
    yva = 1 - yva_true if swapped else yva_true
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xv = torch.tensor(Xva, device=dev)
    n1 = float((ytr == 1).sum())
    n0 = float((ytr == 0).sum())
    pos_weight = torch.tensor([n0 / max(n1, 1)], device=dev)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = SAPLMANet(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(Xt)
    best_score, best_state, best_ep, patience = (-1000000000.0, None, -1, 0)
    history = []
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g, device=dev)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = crit(net(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv)).cpu().numpy()
        auroc = roc_auc_score(yva, pv) if len(set(yva.tolist())) > 1 else 0.5
        ece = _ece(pv, yva.astype(float))
        composite = 0.6 * auroc + 0.4 * (1 - ece)
        history.append({'epoch': ep, 'train_loss': round(tot / n, 4), 'val_auroc': round(float(auroc), 4), 'val_ece': round(ece, 4), 'composite': round(float(composite), 4)})
        if composite > best_score:
            best_score, best_state, best_ep, patience = (composite, copy.deepcopy(net.state_dict()), ep, 0)
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    net.load_state_dict(best_state)
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    out = {'model': model, 'arch': list(ARCH), 'input_dim': int(Xtr.shape[1]), 'best_epoch': best_ep, 'best_composite': round(float(best_score), 4), 'label_mapping': mapping, 'seed': SEED, 'n_train': len(ytr), 'n_val': len(yva), 'eval': {}}
    net.eval()
    for ds in EVAL_SETS:
        X, ids, y_true = load_set(model, ds)
        if X is None:
            continue
        with torch.no_grad():
            p_min = torch.sigmoid(net(torch.tensor(X, device=dev))).cpu().numpy()
        p_correct = 1 - p_min if swapped else p_min
        json.dump({str(i): round(float(p), 5) for i, p in zip(ids, p_correct)}, open(OUT / f'{model}__{ds}.json', 'w'))
        out['eval'][ds] = _metrics(p_correct, y_true)
    torch.save(best_state, WEIGHTS / f'{model}.pth')
    json.dump({**out, 'history': history}, open(OUT / f'{model}__meta.json', 'w'), indent=2)
    ev = out['eval']
    print(f'{model}: best ep {best_ep} composite {best_score:.3f} (swapped={swapped}) | ' + ' '.join((f"{ds.split('_')[0]} AUROC={ev.get(ds, {}).get('auroc')} ECE={ev.get(ds, {}).get('ece')}" for ds in EVAL_SETS)))
    return out

def main(models=None, device='cuda:0'):
    for m in models or MODELS:
        train_one(m, device)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    main([a.model] if a.model else None, a.device)
