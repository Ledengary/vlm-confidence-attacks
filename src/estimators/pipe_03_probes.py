from __future__ import annotations
import argparse
import copy
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from src.config import DATA_DIR, SEED
from src.estimators.metrics_lite import roc_auc_score, ece_score
from src.estimators.train_probe import SAPLMANet, label_mapping
PIPE = DATA_DIR / 'pipeline'
TRAIN, VAL = ('gqa_train_probe_train', 'gqa_train_probe_val')
TEST = ['gqa_val_eval', 'vqav2_ood', 'pope_ood']
CHANNELS = {'saplma': 'saplma', 'pik': 'pik'}
MAX_EPOCHS, PATIENCE, BATCH, LR = (200, 20, 32, 0.001)

def _load_hidden(model, setname, key):
    hid, ids = ([], [])
    for f in sorted(glob.glob(str(PIPE / model / setname / 'shard_*.npz'))):
        d = np.load(f, allow_pickle=True)
        hid.append(d[key].astype(np.float32))
        ids.append(d['ids'])
    if not hid:
        return (None, None)
    return (np.concatenate(hid), np.concatenate(ids))

def _labels(model):
    out = {}
    p = PIPE / model / 'judged.jsonl'
    for l in open(p):
        l = l.strip()
        if l:
            r = json.loads(l)
            out[str(r['item_id'])] = int(r['correct'])
    return out

def _xy(model, setname, key, labels):
    X, ids = _load_hidden(model, setname, key)
    if X is None:
        return (None, None, None)
    keep = [i for i, iid in enumerate(ids) if str(iid) in labels]
    X = X[keep]
    ids = ids[keep]
    y = np.array([labels[str(ids[i])] for i in range(len(ids))], dtype=int)
    return (X, ids, y)

def train_channel(model, key, labels, dev):
    Xtr, _, ytr_true = _xy(model, TRAIN, key, labels)
    Xva, _, yva_true = _xy(model, VAL, key, labels)
    if Xtr is None or Xva is None:
        return None
    swapped, mapping = label_mapping(ytr_true)
    ytr = 1 - ytr_true if swapped else ytr_true
    yva = 1 - yva_true if swapped else yva_true
    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xv = torch.tensor(Xva, device=dev)
    n1 = float((ytr == 1).sum())
    n0 = float((ytr == 0).sum())
    pw = torch.tensor([n0 / max(n1, 1)], device=dev)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = SAPLMANet(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    best_score, best_state, best_ep, patience = (-1000000000.0, None, -1, 0)
    n = len(Xt)
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g, device=dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = crit(net(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv)).cpu().numpy()
        auroc = roc_auc_score(yva, pv) if len(set(yva.tolist())) > 1 else 0.5
        comp = 0.6 * auroc + 0.4 * (1 - ece_score(pv, yva.astype(float)))
        if comp > best_score:
            best_score, best_state, best_ep, patience = (comp, copy.deepcopy(net.state_dict()), ep, 0)
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    net.load_state_dict(best_state)
    out_dir = PIPE / model / 'probe_conf'
    out_dir.mkdir(parents=True, exist_ok=True)
    (PIPE / model / 'probe_weights').mkdir(parents=True, exist_ok=True)
    torch.save(best_state, PIPE / model / 'probe_weights' / f'{key}.pth')
    res = {'channel': key, 'input_dim': int(Xtr.shape[1]), 'best_epoch': best_ep, 'swapped': swapped, 'label_mapping': mapping, 'n_train': len(ytr), 'test': {}}
    net.eval()
    for ds in TEST:
        X, ids, y_true = _xy(model, ds, key, labels)
        if X is None:
            continue
        with torch.no_grad():
            p_min = torch.sigmoid(net(torch.tensor(X, device=dev))).cpu().numpy()
        p_correct = 1 - p_min if swapped else p_min
        json.dump({str(i): round(float(p), 5) for i, p in zip(ids, p_correct)}, open(out_dir / f'{key}__{ds}.json', 'w'))
        auroc = roc_auc_score(y_true, p_correct) if len(set(y_true.tolist())) > 1 else None
        res['test'][ds] = {'n': int(len(y_true)), 'auroc': round(float(auroc), 4) if auroc is not None else None, 'base_rate': round(float(np.mean(y_true)), 4)}
    return res

def run(model, device='cuda:0'):
    dev = device if torch.cuda.is_available() else 'cpu'
    labels = _labels(model)
    out = {'model': model, 'n_labels': len(labels), 'channels': {}}
    for key in CHANNELS.values():
        r = train_channel(model, key, labels, dev)
        if r:
            out['channels'][key] = r
            t = r['test']
            print(f"{model} {key}: best ep {r['best_epoch']} | test AUROC " + ' '.join((f"{ds.split('_')[0]}={t.get(ds, {}).get('auroc')}" for ds in TEST)))
    json.dump(out, open(PIPE / model / 'probes_meta.json', 'w'), indent=2)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    run(a.model, a.device)
