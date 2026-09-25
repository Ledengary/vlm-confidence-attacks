import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
EST_ORDER = ["tokprob", "pik", "saplma", "ccps", "ccpsd", "full", "coupled"]
EST_KEY = {"full": "FULL", "coupled": "COUPLED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t4", required=True)
    ap.add_argument("--floor", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t4 = json.load(open(args.t4))
    prof = {}
    for e in t4:
        if e.get("is_oracle") is not True:
            continue
        prof[(e["model"], e["dataset"], e["evaluated_estimator"])] = (
            e["clean_auroc"], e["witnessed_union_oracle_auroc"]
        )
    floor = json.load(open(args.floor))["floor"]
    rows = []
    for est in EST_ORDER:
        for m in MODEL_ORDER:
            for s in SET_ORDER:
                clean, att = prof[(m, s, est)]
                rows.append({
                    "estimator": EST_KEY.get(est, est),
                    "model": m,
                    "set": s,
                    "clean": clean,
                    "attacked": att,
                    "floor": max(float(floor[m][s]), 0.5),
                })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": rows}, open(args.out, "w"), indent=2)
    print(f"rows {len(rows)}")


if __name__ == "__main__":
    main()
