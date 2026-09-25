import argparse
import glob
import json
import os

import numpy as np

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
OP_ORDER = ["clean10fpr", "youdenJ", "tau0.5"]


def tau_clean_fpr(clean, label, target=0.10):
    wrong = clean[label == 0]
    return 0.5 if len(wrong) == 0 else float(np.quantile(wrong, 1.0 - target))


def tau_youden(clean, label):
    pos = clean[label == 1]
    neg = clean[label == 0]
    best_t, best_j = 0.5, -1.0
    for t in np.unique(clean):
        tpr = float((pos >= t).mean()) if len(pos) else 0.0
        fpr = float((neg >= t).mean()) if len(neg) else 0.0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    return best_t


def cell_stats(clean, adv, label, tau):
    ac = clean >= tau
    aa = adv >= tau
    corr = label == 1
    wrong = label == 0
    nW = int(wrong.sum())
    return {
        "clean_fpr": float((wrong & ac).sum() / nW) if nW else float("nan"),
        "acc_clean": float(label[ac].mean()) if ac.any() else float("nan"),
        "acc_adv": float(label[aa].mean()) if aa.any() else float("nan"),
    }


def load(store, model):
    items = {}
    for f in sorted(glob.glob(os.path.join(store, f"{model}_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            k = (r["dataset"], r["item_id"])
            it = items.get(k)
            if it is None:
                it = {"label": int(r["label"]), "clean_full": float(r["clean_scores"]["full"]), "ep": {}}
                items[k] = it
            if r["objective"] == "full":
                it["ep"][r["direction"]] = float(r["endpoint_scores"]["full"])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--medians", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    med = json.load(open(a.medians))
    percell = {}
    for m in MODEL_ORDER:
        items = load(a.store, m)
        for s in SET_ORDER:
            ci = [it for (ds, iid), it in items.items() if ds == s]
            label = np.array([it["label"] for it in ci], int)
            clean = np.array([it["clean_full"] for it in ci], float)
            mm = med[f"{m}|{s}|full"]
            aB1 = np.array([it["ep"]["min" if it["label"] == 1 else "max"] for it in ci], float)
            aB2 = np.array([it["ep"]["min" if it["clean_full"] >= mm else "max"] for it in ci], float)
            percell[(m, s)] = (clean, label, aB1, aB2)
    out = {"op_order": OP_ORDER, "upper": {}}
    for op in OP_ORDER:
        row = {"clean_fpr": [], "acc_clean": []}
        for cond in ("B1", "B2"):
            acca, drop, below = [], [], 0
            fpr, accc = [], []
            for m in MODEL_ORDER:
                for s in SET_ORDER:
                    clean, label, aB1, aB2 = percell[(m, s)]
                    if op == "clean10fpr":
                        tau = tau_clean_fpr(clean, label, 0.10)
                    elif op == "youdenJ":
                        tau = tau_youden(clean, label)
                    else:
                        tau = 0.5
                    adv = aB1 if cond == "B1" else aB2
                    st = cell_stats(clean, adv, label, tau)
                    fpr.append(st["clean_fpr"]); accc.append(st["acc_clean"])
                    acca.append(st["acc_adv"]); drop.append(st["acc_clean"] - st["acc_adv"])
                    if st["acc_adv"] < 0.5:
                        below += 1
            row["clean_fpr"] = float(np.mean(fpr))
            row["acc_clean"] = float(np.mean(accc))
            row[cond] = {"acc_adv": float(np.mean(acca)), "drop": float(np.mean(drop)),
                         "n_below": below, "n_total": 12}
        out["upper"][op] = row
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
