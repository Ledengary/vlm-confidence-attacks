import argparse
import json
import os
import statistics

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
BETAS = [("0.3", "0p3"), ("1", "1p0"), ("3", "3p0"), ("10", "10p0")]
MODEL_OF = {"internvl": "internvl3_5_2b", "molmo": "molmo2_4b", "llava": "llava_onevision_7b", "gemma": "gemma3_12b"}
SET_OF = {"gqa": "gqa_val_eval", "vqa": "vqav2_ood", "pope": "pope_ood"}
CLEAN_MARGIN = 0.02
CHANCE_BAR = 0.55


def summarize(probes, prior_eval):
    grid = {}
    for lab, tag in BETAS:
        p = probes[f"adv_b{tag}_s23"]
        grid[lab] = {"clean": p["clean"], "attacked": p["attacked"]}
    peak_lab, peak_tag = max(BETAS, key=lambda lt: probes[f"adv_b{lt[1]}_s23"]["attacked"])
    seeds = []
    for sd in (23, 24, 25):
        key = f"adv_b{peak_tag}_s{sd}"
        if key in probes:
            seeds.append(probes[key])
    peak_att_seeds = [x["attacked"] for x in seeds]
    peak_clean = sum(x["clean"] for x in seeds) / len(seeds)
    peak_att = sum(peak_att_seeds) / len(peak_att_seeds)
    clean_thr = prior_eval + CLEAN_MARGIN
    best_attacked = max(v["attacked"] for k, v in probes.items() if k.startswith("adv_b"))
    escape = bool(peak_clean > clean_thr and peak_att > CHANCE_BAR)
    return {
        "grid": grid,
        "peak_beta": peak_lab,
        "peak_att": peak_att,
        "peak_clean": peak_clean,
        "clean_threshold": clean_thr,
        "peak_att_seeds": peak_att_seeds,
        "peak_att_sd": statistics.stdev(peak_att_seeds) if len(peak_att_seeds) > 1 else 0.0,
        "best_attacked": best_attacked,
        "escape": escape,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--breadth", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cells = {m: {} for m in MODEL_ORDER}

    for celltag in sorted(os.listdir(args.breadth)):
        fp = os.path.join(args.breadth, celltag, "frontier.json")
        if not os.path.isfile(fp):
            continue
        with open(fp) as f:
            fj = json.load(f)
        mkey, skey = celltag.split("_", 1)
        m, s = MODEL_OF[mkey], SET_OF[skey]
        rec = summarize(fj["probes"], fj["prior_eval"])
        rec["is_anchor"] = False
        cells[m][s] = rec

    with open(args.anchor) as f:
        anc = json.load(f)
    arec = summarize(anc["probes"], anc["prior_eval"])
    arec["is_anchor"] = True
    cells["llava_onevision_7b"]["vqav2_ood"] = arec
    gradpen = anc["probes"]["gradpen0p1_s23"]

    best_cell = None
    best_val = -1.0
    for m in MODEL_ORDER:
        for s in SET_ORDER:
            v = cells[m][s]["best_attacked"]
            if v > best_val:
                best_val = v
                best_cell = (m, s)

    n_escape = sum(1 for m in MODEL_ORDER for s in SET_ORDER if cells[m][s]["escape"])
    for m in MODEL_ORDER:
        for s in SET_ORDER:
            cells[m][s]["is_nearest_miss"] = bool((m, s) == best_cell)

    nm = cells[best_cell[0]][best_cell[1]]
    out = {
        "cells": cells,
        "beta_labels": [lab for lab, _ in BETAS],
        "n_cells": len(MODEL_ORDER) * len(SET_ORDER),
        "n_escape": n_escape,
        "gradpen_clean": gradpen["clean"],
        "gradpen_attacked": gradpen["attacked"],
        "nearest": {
            "peak_clean": nm["peak_clean"],
            "clean_threshold": nm["clean_threshold"],
            "peak_att": nm["peak_att"],
            "peak_att_sd": nm["peak_att_sd"],
            "peak_att_seeds": nm["peak_att_seeds"],
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out, "n_escape", n_escape, "nearest", best_cell)


if __name__ == "__main__":
    main()
