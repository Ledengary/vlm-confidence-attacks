import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
MODEL_TAG = {
    "internvl3_5_2b": "internvl",
    "molmo2_4b": "molmo",
    "llava_onevision_7b": "llava",
    "gemma3_12b": "gemma",
}
SET_TAG = {"gqa_val_eval": "gqa", "vqav2_ood": "vqa", "pope_ood": "pope"}
KEYS = ["clean", "unb", "att_02", "att_005", "surv_02", "surv_005"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.store, "t4_baseline.json")) as f:
        t4 = json.load(f)
    out = {"cells": {}, "mean": {}}
    acc = {k: [] for k in KEYS}
    for m in MODEL_ORDER:
        out["cells"][m] = {}
        for s in SET_ORDER:
            tag = MODEL_TAG[m] + "_" + SET_TAG[s]
            with open(os.path.join(args.store, tag, "result.json")) as f:
                r = json.load(f)
            clean = r["readouts"]["0.2"]["saplma"]["clean_auroc"]
            att_02 = r["readouts"]["0.2"]["saplma"]["attacked_auroc"]
            att_005 = r["readouts"]["0.05"]["saplma"]["attacked_auroc"]
            unb = t4[tag + "|saplma"]
            surv_02 = (clean - att_02) / (clean - unb)
            surv_005 = (clean - att_005) / (clean - unb)
            cell = {
                "clean": clean,
                "unb": unb,
                "att_02": att_02,
                "att_005": att_005,
                "surv_02": surv_02,
                "surv_005": surv_005,
            }
            out["cells"][m][s] = cell
            for k in KEYS:
                acc[k].append(cell[k])
    for k in KEYS:
        out["mean"][k] = sum(acc[k]) / len(acc[k])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("baked kl_surviving " + str(len(acc["clean"])) + " cells")


if __name__ == "__main__":
    main()
