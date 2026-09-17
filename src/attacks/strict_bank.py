from __future__ import annotations
import json
from src.hidden_states.ccps_features import FEATURE_COLUMNS
from src.estimators.ccpsd import DIFF_FEATURES
from src.attacks.canonical_bank import CCPS_REPAIRED, CanonicalBank
BASE = {'ccps': FEATURE_COLUMNS, 'ccpsd': DIFF_FEATURES}

def check_feature_order(model: str) -> dict:
    out = {}
    for tag in ('ccps', 'ccpsd'):
        d = CCPS_REPAIRED / tag / model / 'best'
        info = json.load(open(d / 'model_info.json'))
        cols = list(info['feature_cols'])
        base = BASE[tag]
        keep = set(cols)
        idx = [i for i, c in enumerate(base) if c in keep]
        selected = [base[i] for i in idx]
        out[tag] = {'n_base': len(base), 'n_selected': len(selected), 'n_declared': len(cols), 'count_matches': len(selected) == len(cols), 'order_matches_exactly': selected == cols, 'first_mismatch': next((j for j in range(min(len(selected), len(cols))) if selected[j] != cols[j]), None), 'keep_idx': idx}
    return out

class StrictCanonicalBank(CanonicalBank):

    def __init__(self, model: str, adapter, device: str, **kw):
        super().__init__(model, adapter, device, **kw)
        for tag in ('ccps', 'ccpsd'):
            cols = getattr(self, f'{tag}_cols', None)
            idx = getattr(self, f'{tag}_keep_idx', None)
            if cols is None or idx is None:
                raise ValueError(f'{tag}/{model}: repaired estimator was not loaded')
            selected = [BASE[tag][i] for i in idx]
            if selected != list(cols):
                j = next((k for k in range(min(len(selected), len(cols))) if selected[k] != cols[k]), min(len(selected), len(cols)))
                raise ValueError(f"{tag}/{model}: selected feature-column ORDER does not equal model_info['feature_cols'] (first mismatch at index {j}: {(selected[j] if j < len(selected) else None)!r} vs {(cols[j] if j < len(cols) else None)!r}). The scaler statistics are stored in feature_cols order, so a permutation would scale every column by the wrong mean and scale without raising.")
        self.strict_feature_order_verified = True
