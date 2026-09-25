import argparse
import csv
import json
import os

import numpy as np

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t4", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t4 = json.load(open(args.t4))
    clean = {}
    oracle = {}
    for e in t4:
        if e.get("is_oracle") is not True:
            continue
        key = (e["model"], e["dataset"])
        clean.setdefault(key, []).append(e["clean_auroc"])
        oracle.setdefault(key, []).append(e["witnessed_union_oracle_auroc"])
    dec = {}
    for r in csv.DictReader(open(args.csv)):
        key = (r["model"], r["dataset"])
        dec.setdefault(key, []).append(r)
    cells = {}
    for m in MODEL_ORDER:
        cells[m] = {}
        for s in SET_ORDER:
            key = (m, s)
            oa = np.array(oracle[key], float)
            ca = np.array(clean[key], float)
            best = max(dec[key], key=lambda r: float(r["adv"]))
            cells[m][s] = {
                "clean_median": float(np.median(ca)),
                "oracle_median": float(np.median(oa)),
                "iqr": float(np.percentile(oa, 75) - np.percentile(oa, 25)),
                "w": float(best["w"]),
                "a_same": float(best["wa_att"]),
                "a_cross": float(best["xa_att"]),
                "a_adv": float(best["adv"]),
                "rho": float(best["rho_pair"]),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"cells": cells}, open(args.out, "w"), indent=2)
    print("cells", sum(len(v) for v in cells.values()))


if __name__ == "__main__":
    main()
