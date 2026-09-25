import argparse
import json
import os
import statistics

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
ESTIMATORS = [
    ("TokProb", "tokprob"),
    ("P(IK)", "pik"),
    ("SAPLMA", "saplma"),
    ("CCPS", "ccps"),
    ("CCPS-D", "ccpsd"),
    ("FULL", "FULL"),
    ("COUPLED", "COUPLED"),
]
DEPLOYED = ["tokprob", "pik", "saplma", "ccps", "ccpsd"]
MODEL_LABEL = {
    "internvl3_5_2b": "InternVL3.5-2B",
    "molmo2_4b": "Molmo2-4B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "gemma3_12b": "Gemma-3-12B",
}
RANDOM_CTRL = {
    "label": "RANDOM (ctrl)",
    "clean_mean": 0.515,
    "clean_median": 0.515,
    "clean_min": 0.515,
    "clean_max": 0.515,
    "adv_mean": 0.513,
    "adv_min": 0.512,
    "adv_max": 0.518,
    "dagger": False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t2 = json.load(open(args.store))["task2"]
    cells = [f"{m}/{s}" for m in MODEL_ORDER for s in SET_ORDER]
    estimators = []
    for label, key in ESTIMATORS:
        clean = [t2[c][key]["clean"]["within_auroc"] for c in cells]
        adv = [t2[c][key]["adv"]["within_auroc"] for c in cells]
        estimators.append({
            "label": label,
            "clean_mean": statistics.mean(clean),
            "clean_median": statistics.median(clean),
            "clean_min": min(clean),
            "clean_max": max(clean),
            "adv_mean": statistics.mean(adv),
            "adv_min": min(adv),
            "adv_max": max(adv),
            "dagger": label == "COUPLED",
        })
    estimators.append(RANDOM_CTRL)
    per_model = []
    for m in MODEL_ORDER:
        vals = [t2[f"{m}/{s}"][k]["adv"]["within_auroc"] for s in SET_ORDER for k in DEPLOYED]
        note = "(partial inversion)" if m == "molmo2_4b" else ""
        per_model.append({"label": MODEL_LABEL[m], "auroc": statistics.mean(vals), "note": note})
    coupled_cells = {c: t2[c]["COUPLED"]["adv"]["within_auroc"] for c in cells}
    molmo_pope = coupled_cells["molmo2_4b/pope_ood"]
    twelve = statistics.mean(list(coupled_cells.values()))
    eleven = statistics.mean([v for c, v in coupled_cells.items() if c != "molmo2_4b/pope_ood"])
    out = {
        "estimators": estimators,
        "per_model": per_model,
        "coupled_dagger": {"twelve_cell": twelve, "molmo_pope": molmo_pope, "eleven_cell": eleven},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"within_answer: {len(estimators)} estimator rows, {len(per_model)} model rows")


if __name__ == "__main__":
    main()
