from __future__ import annotations
import numpy as np

def auroc(scores, y):
    s = np.asarray(scores, float)
    y = np.asarray(y)
    pos, neg = (s[y == 1], s[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a, b = (pos[:, None], neg[None, :])
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (len(pos) * len(neg)))

def witnessed_union_scores(per_item_witnessed):
    a = np.asarray(per_item_witnessed, float)
    return (a.min(axis=1), a.max(axis=1))

def sel_witnessed_union_oracle(per_item_witnessed, y):
    lo, hi = witnessed_union_scores(per_item_witnessed)
    return np.where(np.asarray(y) == 1, lo, hi).astype(float)
