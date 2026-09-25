import argparse
import glob
import json
import os

import numpy as np

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]


def tau_clean_fpr(clean, label, target=0.10):
    wrong = clean[label == 0]
    return 0.5 if len(wrong) == 0 else float(np.quantile(wrong, 1.0 - target))


def flips(clean, adv, label, tau):
    ac = clean >= tau
    aa = adv >= tau
    corr = label == 1
    wrong = label == 0
    nW = int(wrong.sum())
    return {
        "w2ac": float((wrong & ~ac & aa).sum() / nW) if nW else float("nan"),
        "acc_clean": float(label[ac].mean()) if ac.any() else float("nan"),
        "acc_adv": float(label[aa].mean()) if aa.any() else float("nan"),
    }


def reprior(a, p):
    return (a * p) / (a * p + (1.0 - a) * (1.0 - p))


def aurc(scores, label):
    s = np.asarray(scores, float)
    y = np.asarray(label, float)
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    cw = np.arange(1, len(y) + 1, dtype=float)
    risk = np.cumsum(1.0 - yy) / cw
    cov = cw / cw[-1]
    cov_prev = np.concatenate([[0.0], cov[:-1]])
    return float(np.sum((cov - cov_prev) * risk))


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
            o = r["objective"]
            if o in ("full", "pik"):
                it["ep"][(o, r["direction"])] = float(r["endpoint_scores"]["full"])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--natprev", required=True)
    ap.add_argument("--medians", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    natprev = json.load(open(a.natprev))["cells"]
    med = json.load(open(a.medians))
    out = {"cells": {}}
    for m in MODEL_ORDER:
        items = load(a.store, m)
        out["cells"][m] = {}
        for s in SET_ORDER:
            ci = [it for (ds, iid), it in items.items() if ds == s]
            label = np.array([it["label"] for it in ci], int)
            clean = np.array([it["clean_full"] for it in ci], float)
            p = natprev[f"{m}/{s}/FULL"]["p_natural"]
            mm = med[f"{m}|{s}|full"]
            tau = tau_clean_fpr(clean, label, 0.10)

            def adv(aim, rule):
                arr = []
                for it in ci:
                    if rule == "oracle":
                        d = "min" if it["label"] == 1 else "max"
                    else:
                        d = "min" if it["clean_full"] >= mm else "max"
                    arr.append(it["ep"][(aim, d)])
                return np.array(arr, float)

            adv_b1 = adv("full", "oracle"); adv_b2 = adv("full", "lf")
            b1 = flips(clean, adv_b1, label, tau)
            b2 = flips(clean, adv_b2, label, tau)
            b3 = flips(clean, adv("pik", "oracle"), label, tau)
            out["cells"][m][s] = {
                "acc_clean": b1["acc_clean"],
                "w2ac_b1": b1["w2ac"], "acc_adv_b1": b1["acc_adv"],
                "w2ac_b2": b2["w2ac"], "acc_adv_b2": b2["acc_adv"],
                "w2ac_b3": b3["w2ac"], "acc_adv_b3": b3["acc_adv"],
                "p_natural": p,
                "rew_adv_b1": reprior(b1["acc_adv"], p),
                "rew_adv_b2": reprior(b2["acc_adv"], p),
                "rew_adv_b3": reprior(b3["acc_adv"], p),
                "below_b1": bool(reprior(b1["acc_adv"], p) < p),
                "below_b2": bool(reprior(b2["acc_adv"], p) < p),
                "below_b3": bool(reprior(b3["acc_adv"], p) < p),
                "aurc_clean": aurc(clean, label),
                "aurc_b1": aurc(adv_b1, label),
                "aurc_b2": aurc(adv_b2, label),
            }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
