import argparse
import csv
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
EST_ORDER = ["tokprob", "pik", "saplma", "ccps", "ccpsd", "full", "coupled"]
EST_KEY = {"full": "FULL", "coupled": "COUPLED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = {}
    for r in csv.DictReader(open(args.csv)):
        table[(r["model"], r["dataset"], r["estimator"])] = {
            "w": float(r["w"]),
            "a_same": float(r["wa_att"]),
            "a_cross": float(r["xa_att"]),
            "a_adv": float(r["adv"]),
            "rho": float(r["rho_pair"]),
        }
    rows = []
    for m in MODEL_ORDER:
        for s in SET_ORDER:
            for e in EST_ORDER:
                v = table[(m, s, e)]
                rows.append({
                    "model": m,
                    "set": s,
                    "estimator": EST_KEY.get(e, e),
                    "w": v["w"],
                    "a_same": v["a_same"],
                    "a_cross": v["a_cross"],
                    "a_adv": v["a_adv"],
                    "rho": v["rho"],
                })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": rows}, open(args.out, "w"), indent=2)
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()
