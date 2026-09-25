from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json

import numpy as np
import torch
import torch.nn as nn

from src.config import DATA_DIR, SEED
from src.estimators.train_probe import SAPLMANet, label_mapping

OUT = DATA_DIR / "long_answer"
MAX_EPOCHS, PATIENCE, BATCH, LR = 200, 20, 32, 1e-3


def ece_score(p, y, bins=15):
    p = np.asarray(p)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(p)
    return e


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return 0.5
    o = np.argsort(s, kind="mergesort")
    rk = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[o[j + 1]] == s[o[i]]:
            j += 1
        rk[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def img_hash(png_rel):
    return hashlib.sha256(open(DATA_DIR / png_rel, "rb").read()).hexdigest()


def load_cell(model):
    cell = OUT / model
    H = {}
    for f in glob.glob(str(cell / "hidden_s*.npz")):
        z = np.load(f, allow_pickle=True)
        for k, i in enumerate(z["ids"]):
            hp = z["h_pik"][k].astype(np.float16).astype(np.float32)
            hs = z["h_sap"][k].astype(np.float16).astype(np.float32)
            H[str(i)] = (hp, hs)
    lab = {}
    for l in open(cell / "labels.jsonl"):
        if l.strip():
            r = json.loads(l)
            lab[r["item_id"]] = r["label"]
    dec = {}
    for f in glob.glob(str(cell / "decode_s*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if "ERROR" not in r:
                    dec[r["item_id"]] = r
    ids = sorted(i for i in H if i in lab and i in dec)
    return H, lab, dec, ids


def image_disjoint_split(ids, dec, eval_frac=0.4):
    groups = {i: img_hash(dec[i]["clean_png"]) for i in ids}
    uniq = sorted(set(groups.values()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    target = int(round(eval_frac * len(ids)))
    ev_g, n = set(), 0
    for g in uniq:
        if n >= target:
            break
        ev_g.add(g)
        n += sum(1 for i in ids if groups[i] == g)
    ev = [i for i in ids if groups[i] in ev_g]
    tr = [i for i in ids if groups[i] not in ev_g]
    assert not (set(groups[i] for i in tr) & set(groups[i] for i in ev)), "image leakage"
    return tr, ev, len(set(groups.values()))


def train_probe(X, y_true, dev):
    swapped, mapping = label_mapping(y_true)
    y = (1 - y_true) if swapped else y_true
    Xt = torch.tensor(X, device=dev, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    pw = torch.tensor([n0 / max(n1, 1)], device=dev)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = SAPLMANet(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    g = torch.Generator(device=dev)
    g.manual_seed(SEED)
    n = len(Xt)
    nval = max(1, int(0.25 * n))
    idx = torch.randperm(n, generator=g, device=dev)
    vidx, tidx = idx[:nval], idx[nval:]
    best, best_state, pat = -1e9, None, 0
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = tidx[torch.randperm(len(tidx), generator=g, device=dev)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = crit(net(Xt[b]), yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xt[vidx])).cpu().numpy()
        yv = y[vidx.cpu().numpy()]
        comp = 0.6 * auroc(yv, pv) + 0.4 * (1 - ece_score(pv, yv.astype(float)))
        if comp > best:
            best, best_state, pat = comp, copy.deepcopy(net.state_dict()), 0
        else:
            pat += 1
        if pat >= PATIENCE:
            break
    net.load_state_dict(best_state)
    net.cpu().eval()
    return net, swapped, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    H, lab, dec, ids = load_cell(a.model)
    tr, ev, nimg = image_disjoint_split(ids, dec)
    print(f"[{a.model}] items={len(ids)} images={nimg} train={len(tr)} eval={len(ev)} "
          f"train_acc={np.mean([lab[i] for i in tr]):.3f} eval_acc={np.mean([lab[i] for i in ev]):.3f}")
    out = {"split": {"train": tr, "eval": ev, "n_images": nimg}, "probes": {}}
    for key, locus in (("pik", 0), ("saplma", 1)):
        Xtr = np.stack([H[i][locus] for i in tr])
        ytr = np.array([lab[i] for i in tr])
        net, swapped, mapping = train_probe(Xtr, ytr, dev)
        Xev = torch.tensor(np.stack([H[i][locus] for i in ev]), dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(net(Xev)).numpy()
        pcorr = (1 - p) if swapped else p
        yev = np.array([lab[i] for i in ev])
        out["probes"][key] = {"state": net.state_dict(), "swapped": swapped, "mapping": mapping,
                              "input_dim": int(Xtr.shape[1]),
                              "eval_clean_auroc": round(float(auroc(yev, pcorr)), 4)}
        print(f"  {key}: swapped={swapped} eval_clean_AUROC={out['probes'][key]['eval_clean_auroc']}")
    torch.save(out, OUT / a.model / "probes.pt")
    json.dump({"train": tr, "eval": ev, "n_images": nimg,
               "pik_clean_auroc": out["probes"]["pik"]["eval_clean_auroc"],
               "saplma_clean_auroc": out["probes"]["saplma"]["eval_clean_auroc"]},
              open(OUT / a.model / "split.json", "w"), indent=2)


if __name__ == "__main__":
    main()
