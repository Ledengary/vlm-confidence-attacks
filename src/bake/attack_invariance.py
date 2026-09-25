import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
COORD = ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "D1", "D2"]
LABELFREE = ["C1", "C2"]


def load_cells(gen104, name):
    with open(os.path.join(gen104, f"{name}.json")) as f:
        return json.load(f)["cells"]


def coord_pass(cell):
    u = cell["axes"]["u_logreg"]
    return u["harmful_frac"]["mean"] > 0.5 and u["meandir_cos"] > u["null_abs_p975"] and u["emp_p"] < 0.05


def labelfree_pass(cell):
    u = cell["uncoord"]["u_logreg"]
    return bool(u["sign_ok"] and abs(u["meandir_cos"]) > u["null_abs_p975"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen104", required=True)
    ap.add_argument("--matchedmargin", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    coord = {c: load_cells(args.gen104, c) for c in COORD}
    lf = {c: load_cells(args.gen104, c) for c in LABELFREE}

    block_a = {"cells": {}}
    total_coord = 0
    total_lf = 0
    lf_failures = []
    for c in LABELFREE:
        for m in MODEL_ORDER:
            for s in SET_ORDER:
                if not labelfree_pass(lf[c][f"{m}/{s}/sap"]):
                    lf_failures.append({"config": c, "model": m, "set": s})
    for m in MODEL_ORDER:
        block_a["cells"][m] = {}
        for s in SET_ORDER:
            key = f"{m}/{s}/sap"
            cp = sum(1 for c in COORD if coord_pass(coord[c][key]))
            lp = sum(1 for c in LABELFREE if labelfree_pass(lf[c][key]))
            block_a["cells"][m][s] = {
                "coord_pass": cp,
                "coord_total": len(COORD),
                "lf_pass": lp,
                "lf_total": len(LABELFREE),
            }
            total_coord += cp
            total_lf += lp

    with open(args.matchedmargin) as f:
        mm = json.load(f)["cells"]
    block_b = {"cells": {}}
    n_below_floor = 0
    n_below_half = 0
    above_half = []
    for m in MODEL_ORDER:
        block_b["cells"][m] = {}
        for s in SET_ORDER:
            ub = mm[f"saplma/unbounded/{m}/{s}"]
            bd = mm[f"saplma/0.25/{m}/{s}"]
            clean = ub["clean_auroc_engaged"]
            adv_u = ub["adv_auroc_engaged"]
            adv_m = bd["adv_auroc_engaged"]
            floor = ub["floor"]
            block_b["cells"][m][s] = {
                "clean": clean,
                "adv_unbounded": adv_u,
                "adv_margin": adv_m,
                "floor": floor,
            }
            if adv_m <= floor:
                n_below_floor += 1
            if adv_m < 0.5:
                n_below_half += 1
            else:
                above_half.append({"model": m, "set": s, "value": adv_m})

    out = {
        "block_a": block_a,
        "total_coord": total_coord,
        "total_coord_max": len(COORD) * len(MODEL_ORDER) * len(SET_ORDER),
        "total_lf": total_lf,
        "total_lf_max": len(LABELFREE) * len(MODEL_ORDER) * len(SET_ORDER),
        "lf_failures": lf_failures,
        "block_b": block_b,
        "n_cells": len(MODEL_ORDER) * len(SET_ORDER),
        "n_below_floor": n_below_floor,
        "n_below_half": n_below_half,
        "above_half": above_half,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out, "coord", total_coord, "lf", total_lf, "failures", len(lf_failures))


if __name__ == "__main__":
    main()
