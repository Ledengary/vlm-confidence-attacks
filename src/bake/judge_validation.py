import argparse
import json
import os

MODEL_MAP = {
    "InternVL3.5-2B": "internvl3_5_2b",
    "Molmo2-4B": "molmo2_4b",
    "LLaVA-OneVision-7B": "llava_onevision_7b",
    "Gemma-3-12B": "gemma3_12b",
}
MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["GQA", "VQAv2", "POPE"]


def kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for k in (0, 1):
        pa = sum(1 for x in a if x == k) / n
        pb = sum(1 for x in b if x == k) / n
        pe += pa * pb
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(rows):
    return sum(1 for r in rows if r["judge_label"] == r["my_label"]) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ci", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.labels) as f:
        d = json.load(f)
    with open(args.ci) as f:
        ci = json.load(f)["overall"]["kappa_ci"]
    jl = [r["judge_label"] for r in d]
    ml = [r["my_label"] for r in d]
    out = {
        "overall": {
            "n": len(d),
            "agreement": agreement(d),
            "kappa": kappa(jl, ml),
            "kappa_ci": [float(ci[0]), float(ci[1])],
        },
        "per_dataset": {},
        "per_model": {},
    }
    for ds in SET_ORDER:
        sub = [r for r in d if r["dataset"] == ds]
        out["per_dataset"][ds] = {"n": len(sub), "agreement": agreement(sub)}
    by_key = {}
    for r in d:
        by_key.setdefault(MODEL_MAP[r["model"]], []).append(r)
    for m in MODEL_ORDER:
        sub = by_key.get(m, [])
        if not sub:
            continue
        out["per_model"][m] = {"n": len(sub), "agreement": agreement(sub)}
    wrong = [r for r in d if r["my_label"] == 0]
    out["wrong"] = {
        "n": len(wrong),
        "n_agree": sum(1 for r in wrong if r["judge_label"] == 0),
        "false_accept": sum(1 for r in wrong if r["judge_label"] == 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"overall agreement {out['overall']['agreement']:.3f} kappa {out['overall']['kappa']:.4f}")


if __name__ == "__main__":
    main()
