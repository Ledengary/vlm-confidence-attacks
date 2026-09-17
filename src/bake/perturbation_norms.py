import argparse
import glob
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    agg = {m: {s: {"sum": 0.0, "max": 0.0, "n_moved": 0, "n_total": 0} for s in SET_ORDER} for m in MODEL_ORDER}
    for path in sorted(glob.glob(os.path.join(args.runs, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                m, s = r.get("model"), r.get("dataset")
                if m not in agg or s not in agg[m]:
                    continue
                v = r.get("validity") or {}
                agg[m][s]["n_total"] += 1
                if v.get("endpoint_is_clean_image") is True:
                    continue
                linf = v.get("linf_raw")
                if linf is None:
                    continue
                linf = float(linf)
                agg[m][s]["sum"] += linf
                agg[m][s]["n_moved"] += 1
                if linf > agg[m][s]["max"]:
                    agg[m][s]["max"] = linf
    out = {"cells": {}, "n_moved_total": 0, "n_total": 0}
    for m in MODEL_ORDER:
        out["cells"][m] = {}
        for s in SET_ORDER:
            a = agg[m][s]
            mean = a["sum"] / a["n_moved"] if a["n_moved"] else 0.0
            out["cells"][m][s] = {
                "linf_mean": mean,
                "linf_max": a["max"],
                "n_moved": a["n_moved"],
                "n_total": a["n_total"],
            }
            out["n_moved_total"] += a["n_moved"]
            out["n_total"] += a["n_total"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"moved {out['n_moved_total']} of {out['n_total']}")


if __name__ == "__main__":
    main()
