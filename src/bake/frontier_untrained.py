import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
INFORMATIVE = ["tokprob", "pik", "saplma", "ccps", "ccpsd", "full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t4", required=True)
    ap.add_argument("--t6", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t4 = json.load(open(args.t4))
    by = {}
    for e in t4:
        if e.get("is_oracle") is True:
            by[(e["model"], e["dataset"], e["evaluated_estimator"])] = (
                e["clean_auroc"], e["witnessed_union_oracle_auroc"]
            )
    points = []
    for m in MODEL_ORDER:
        for s in SET_ORDER:
            cl = sum(by[(m, s, e)][0] for e in INFORMATIVE) / len(INFORMATIVE)
            at = sum(by[(m, s, e)][1] for e in INFORMATIVE) / len(INFORMATIVE)
            points.append({"model": m, "set": s, "clean": cl, "attacked": at})
    t6 = json.load(open(args.t6))
    diag = [e["auroc"] for e in t6 if e.get("selector") == "bqa_selected"
            and e.get("aimed_objective") == e.get("evaluated_estimator")
            and isinstance(e.get("auroc"), (int, float))]
    b_qa = sum(diag) / len(diag)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"points": points, "b_qa": b_qa}, open(args.out, "w"), indent=2)
    print("points", len(points), "b_qa", round(b_qa, 3))


if __name__ == "__main__":
    main()
