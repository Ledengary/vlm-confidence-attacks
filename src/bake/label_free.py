import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_STORE = os.path.join(REPO_ROOT, "artifacts", "labelfree_clean_sign_all.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "artifacts", "label_free.json")

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
CHANNELS = ["tokprob", "pik", "saplma", "ccps", "ccpsd", "full", "coupled"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    with open(args.store) as f:
        recs = json.load(f)
    lut = {}
    for r in recs:
        if r.get("selector") != "clean_sign_selected":
            continue
        if r.get("is_oracle") is not False:
            continue
        if r.get("aimed_objective") != r.get("evaluated_estimator"):
            continue
        est = r.get("evaluated_estimator")
        if est not in CHANNELS:
            continue
        lut[(r.get("model"), r.get("dataset"), est)] = float(r.get("auroc"))
    out = {"cells": {}}
    n = 0
    for m in MODEL_ORDER:
        out["cells"][m] = {}
        for s in SET_ORDER:
            cell = {}
            for est in CHANNELS:
                cell[est] = lut[(m, s, est)]
                n += 1
            out["cells"][m][s] = cell
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {n} values across {len(MODEL_ORDER)} models x {len(SET_ORDER)} sets")


if __name__ == "__main__":
    main()
