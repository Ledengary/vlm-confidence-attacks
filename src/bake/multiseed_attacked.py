import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
SEEDS = ["23", "24", "25"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.store) as f:
        ms = json.load(f)
    by_cell = {c["cell"]: c for c in ms["cells"]}
    out = {"cells": {}}
    for m in MODEL_ORDER:
        out["cells"][m] = {}
        for s in SET_ORDER:
            c = by_cell[f"{m}/{s}"]
            seeds = {}
            for sd in SEEDS:
                e = c["seeds"][sd]
                a = float(e["attacked_auroc_full"]) if "attacked_auroc_full" in e else float(e["attacked_auroc"])
                seeds[sd] = {"a": a, "ci": [float(e["ci"][0]), float(e["ci"][1])]}
            out["cells"][m][s] = {
                "readout": c["seeds"]["23"]["binding_readout"],
                "clean": float(c["clean_mean"]),
                "seeds": seeds,
                "mean": float(c.get("attacked_mean_full", c["attacked_mean"])),
                "sd": float(c.get("attacked_sd_full", c["attacked_sd"])),
                "survivor": not bool(c["all_below_chance"]),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    n = sum(1 for m in MODEL_ORDER for s in SET_ORDER)
    print(f"wrote {n} cells across {len(MODEL_ORDER)} models x {len(SET_ORDER)} sets")


if __name__ == "__main__":
    main()
