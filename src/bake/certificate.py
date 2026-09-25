import argparse
import json
import os

CHANNELS = ["tokprob", "pik", "saplma", "ccps", "ccpsd", "FULL", "COUPLED"]
MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    store = json.load(open(args.store))["cells"]
    rows = []
    for ch in CHANNELS:
        for m in MODEL_ORDER:
            for s in SET_ORDER:
                e = store[f"{m}/{s}"][ch]
                rows.append({
                    "channel": ch,
                    "model": m,
                    "set": s,
                    "gamma_grid": e["gamma_star"],
                    "gamma_exact": e["gamma_star_exact"],
                    "ci": e["gamma_star_ci"],
                    "sep": e["separation"]["mean"],
                    "g0": e["clean_auroc"],
                    "disp": e["delta_aimed_feasible"],
                    "wmax": e["witness_max"],
                    "nfeas": e["n_feasible"],
                    "frac": e["frac_exceed_gamma_exact"],
                    "floor": e["floor"],
                })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": rows}, open(args.out, "w"), indent=2)
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()
