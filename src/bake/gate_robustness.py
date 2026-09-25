import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
OP_ORDER = ["clean10fpr", "youdenJ", "tau0.5"]


def mean(xs):
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", required=True)
    ap.add_argument("--t10", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    triage = {}
    for m in MODEL_ORDER:
        with open(os.path.join(args.triage, m, "triage.json")) as f:
            triage[m] = json.load(f)

    op_block = {}
    for op in OP_ORDER:
        fpr, accc, acca, drop = [], [], [], []
        below = 0
        for m in MODEL_ORDER:
            for s in SET_ORDER:
                c = triage[m]["cells"][f"{s}/FULL"]["operating_points"][op]
                fpr.append(c["clean_fpr_realized"])
                accc.append(c["gate_acc_clean"])
                acca.append(c["gate_acc_adv"])
                drop.append(c["gate_acc_clean"] - c["gate_acc_adv"])
                if c["gate_acc_adv"] < 0.5:
                    below += 1
        op_block[op] = {
            "clean_fpr": mean(fpr),
            "acc_clean": mean(accc),
            "acc_adv": mean(acca),
            "drop": mean(drop),
            "n_below": below,
            "n_total": len(acca),
        }

    with open(args.t10) as f:
        t10 = json.load(f)
    full_rows = {
        (r["model"], r["dataset"]): r
        for r in t10
        if r.get("aimed_objective") == "full"
        and r.get("evaluated_estimator") == "full"
        and r.get("diagonal")
    }
    aurc = {"cells": {}}
    fc, fa = [], []
    for m in MODEL_ORDER:
        aurc["cells"][m] = {}
        for s in SET_ORDER:
            r = full_rows[(m, s)]
            aurc["cells"][m][s] = {
                "clean": r["clean_aurc_riemann"],
                "attacked": r["attacked_aurc_riemann"],
            }
            fc.append(r["clean_aurc_riemann"])
            fa.append(r["attacked_aurc_riemann"])
    diag = [
        r
        for r in t10
        if r.get("diagonal") and r.get("aimed_objective") == r.get("evaluated_estimator")
    ]

    out = {
        "operating_points": op_block,
        "op_order": OP_ORDER,
        "aurc": aurc,
        "full_mean_clean": mean(fc),
        "full_mean_attacked": mean(fa),
        "diag60_clean": mean([r["clean_aurc_riemann"] for r in diag]),
        "diag60_attacked": mean([r["attacked_aurc_riemann"] for r in diag]),
        "n_diag": len(diag),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
