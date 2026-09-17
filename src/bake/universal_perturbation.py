import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

MODELS = ["gemma3_12b", "llava_onevision_7b"]
SETS = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
CHANS = [("tok", "token-prob"), ("pik", "P(IK)"), ("sap", "SAPLMA"), ("verb", "verbalized"),
         ("sc", "self-consist."), ("ccps", "CCPS"), ("ccpsd", "CCPS-D")]
SEED = 23
N_BOOT = 10000
BLOCK_B_MODEL = "gemma3_12b"
BLOCK_B_SET = "gqa_val_eval"


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    p = int((y == 1).sum())
    n = int((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def load_cell(univ_dir, model, dataset):
    rows = []
    for f in sorted(glob.glob(os.path.join(univ_dir, model, "eval", "s*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("dataset") == dataset:
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universal-dir", required=True)
    ap.add_argument("--floor-bound", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tfb = json.load(open(args.floor_bound))["cells"]
    block_a = {}
    for m in MODELS:
        block_a[m] = {}
        for s in SETS:
            rows = load_cell(args.universal_dir, m, s)
            y = np.array([d["label"] for d in rows])
            maxdiff = 0.0
            for ck, _ in CHANS:
                cl = [d["clean"].get(ck) for d in rows]
                un = [d["univ"].get(ck) for d in rows]
                rn = [d["rand"].get(ck) for d in rows]
                ud = auc(y, cl) - auc(y, un)
                rd = auc(y, cl) - auc(y, rn)
                maxdiff = max(maxdiff, abs(ud - rd))
            sap = tfb[m][s]["saplma"]
            pim = sap["clean_auroc"] - sap["adv_auroc"]
            block_a[m][s] = {"maxdiff": maxdiff, "pim": pim}

    rows = load_cell(args.universal_dir, BLOCK_B_MODEL, BLOCK_B_SET)
    y = np.array([d["label"] for d in rows])
    rng = np.random.default_rng(SEED)
    idxb = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    channels = []
    for ck, cn in CHANS:
        cl = np.array([d["clean"].get(ck, np.nan) for d in rows])
        un = np.array([d["univ"].get(ck, np.nan) for d in rows])
        rn = np.array([d["rand"].get(ck, np.nan) for d in rows])
        clean = auc(y, cl)
        ud = clean - auc(y, un)
        rd = clean - auc(y, rn)
        diff = ud - rd
        db = np.empty(N_BOOT)
        for b, ix in enumerate(idxb):
            db[b] = (auc(y[ix], cl[ix]) - auc(y[ix], un[ix])) - (auc(y[ix], cl[ix]) - auc(y[ix], rn[ix]))
        lo, hi = float(np.nanpercentile(db, 2.5)), float(np.nanpercentile(db, 97.5))
        channels.append({"name": cn, "clean": float(clean), "uni_drop": float(ud),
                         "rand_drop": float(rd), "diff": float(diff), "ci": [lo, hi]})

    out = {
        "models": MODELS,
        "sets": SETS,
        "block_a": block_a,
        "block_b": {"model": BLOCK_B_MODEL, "set": BLOCK_B_SET, "channels": channels},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} with {len(MODELS)} models and {len(channels)} channels")


if __name__ == "__main__":
    main()
