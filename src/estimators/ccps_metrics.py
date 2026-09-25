from __future__ import annotations
import numpy as np
from scipy.stats import rankdata

def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def auprc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    npos = int((y == 1).sum())
    if npos == 0:
        return float('nan')
    order = np.argsort(-s, kind='mergesort')
    yo = y[order]
    tp = np.cumsum(yo)
    fp = np.cumsum(1 - yo)
    prec = tp / (tp + fp)
    rec = tp / npos
    ap = 0.0
    prev = 0.0
    for i in range(len(yo)):
        ap += (rec[i] - prev) * prec[i]
        prev = rec[i]
    return float(ap)

def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float('nan')

def ece(y, p, bins=15):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return float('nan')
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return e
