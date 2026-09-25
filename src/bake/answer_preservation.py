import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
POOL_SIZE = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigor", required=True)
    ap.add_argument("--cstar", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.rigor) as f:
        guarantees = json.load(f)["guarantees"]
    with open(args.cstar) as f:
        cstar = json.load(f)["cstar"]
    out = {"cells": {}}
    for m in MODEL_ORDER:
        out["cells"][m] = {}
        for s in SET_ORDER:
            n, ok, rate = guarantees[f"{m}|{s}"]
            bid = int(cstar[m][s]["n_byte_identical"])
            out["cells"][m][s] = {
                "rigor_n": int(n),
                "rigor_rate": float(rate),
                "byte_id": bid,
                "byte_total": POOL_SIZE,
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(MODEL_ORDER) * len(SET_ORDER)} cells")


if __name__ == "__main__":
    main()
