import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]


def reprior(acc, p):
    return (acc * p) / (acc * p + (1.0 - acc) * (1.0 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triage", required=True)
    ap.add_argument("--natprev", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.natprev) as f:
        natprev = json.load(f)["cells"]
    out = {"cells": {}}
    for m in MODEL_ORDER:
        with open(os.path.join(args.triage, m, "triage.json")) as f:
            d = json.load(f)
        out["cells"][m] = {}
        for s in SET_ORDER:
            trans = d["cells"][f"{s}/FULL"]["operating_points"]["clean10fpr"]
            aimed = d["adaptive_cells"][f"{s}/FULL/main"]["operating_points"]["clean10fpr"]
            p = natprev[f"{m}/{s}/FULL"]["p_natural"]
            cl_rw = reprior(trans["gate_acc_clean"], p)
            tr_rw = reprior(trans["gate_acc_adv"], p)
            ai_rw = reprior(aimed["gate_acc_adv"], p)
            out["cells"][m][s] = {
                "w2ac": trans["w2ac"],
                "acc_clean": trans["gate_acc_clean"],
                "acc_adv": trans["gate_acc_adv"],
                "w2ac_aimed": aimed["w2ac"],
                "acc_adv_aimed": aimed["gate_acc_adv"],
                "p_natural": p,
                "rew_clean": cl_rw,
                "rew_adv_transfer": tr_rw,
                "rew_adv_aimed": ai_rw,
                "below_transfer": bool(tr_rw < p),
                "below_aimed": bool(ai_rw < p),
            }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
