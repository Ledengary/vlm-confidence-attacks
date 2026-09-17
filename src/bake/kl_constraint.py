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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"readout": "saplma", "cells": {}}
    for m in MODEL_ORDER:
        out["cells"][m] = {}
        for s in SET_ORDER:
            tag = MODEL_TAG[m] + "_" + SET_TAG[s]
            with open(os.path.join(args.store, tag, "result.json")) as f:
                r = json.load(f)
            r02 = r["readouts"]["0.2"]["saplma"]
            r005 = r["readouts"]["0.05"]["saplma"]
            out["cells"][m][s] = {
                "clean": r02["clean_auroc"],
                "att_02": r02["attacked_auroc"],
                "att_005": r005["attacked_auroc"],
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("baked kl_constraint " + str(len(MODEL_ORDER) * len(SET_ORDER)) + " cells")


if __name__ == "__main__":
    main()
