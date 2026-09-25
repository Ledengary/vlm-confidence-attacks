from __future__ import annotations
import numpy as np

def _rankdata(a):
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1)
    a_sorted = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks

def roc_auc_score(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = _rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def average_precision_score(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    denom = tp + fp
    P = np.where(denom > 0, tp / denom, 0.0)
    total_pos = max(float(y.sum()), 1.0)
    R = tp / total_pos
    R_prev = np.concatenate([[0.0], R[:-1]])
    return float(np.sum((R - R_prev) * P))

def brier_score_loss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0, 1)
    return float(np.mean((p - y) ** 2))

def ece_score(p, y, bins=10):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

class LogisticCombiner:

    def __init__(self, max_iter=2000, l2=0.001, seed=23):
        self.max_iter, self.l2, self.seed = (max_iter, l2, seed)

    def fit(self, X, y):
        import torch
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-08
        Xs = (X - self.mu) / self.sd
        torch.manual_seed(self.seed)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.lin = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(self.lin.parameters(), lr=0.05, weight_decay=self.l2)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.max_iter):
            opt.zero_grad()
            loss_fn(self.lin(Xt).squeeze(-1), yt).backward()
            opt.step()
        return self

    def predict_proba(self, X):
        import torch
        Xs = (np.asarray(X, float) - self.mu) / self.sd
        with torch.no_grad():
            p = torch.sigmoid(self.lin(torch.tensor(Xs, dtype=torch.float32)).squeeze(-1)).numpy()
        return np.column_stack([1 - p, p])
