import argparse
import csv
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amplitude-store", required=True)
    ap.add_argument("--floor-store", required=True)
    ap.add_argument("--ladder-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    amp = json.load(open(args.amplitude_store))
    cv5 = {}
    for r in amp["rows"]:
        cv5[(r["model"], r["dataset"])] = r["floor_raw_cv5"]
    fb = json.load(open(args.floor_store))["floor"]
    rows = list(csv.DictReader(open(args.ladder_csv)))
    cells = []
    for m in MODEL_ORDER:
        for s in SET_ORDER:
            group = [r for r in rows if r["model"] == m and r["dataset"] == s]
            binding = max(group, key=lambda r: float(r["adv"]))
            heldout = fb[m]["gqa_val_eval"] if s == "gqa_val_eval" else None
            cells.append({
                "model": m,
                "set": s,
                "cv5": cv5[(m, s)],
                "heldout": heldout,
                "operative": max(fb[m][s], 0.5),
                "rho": float(binding["rho_pair"]),
                "w": float(binding["w"]),
                "cstar": float(binding["c_star"]),
                "ceiling": float(binding["ceiling"]),
                "holds": all(r["holds"] == "True" for r in group),
            })
    out = {"cells": cells}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"ceiling_floor: {len(cells)} cells, holds {sum(c['holds'] for c in cells)}/{len(cells)}")


if __name__ == "__main__":
    main()
