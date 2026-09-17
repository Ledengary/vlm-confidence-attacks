import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
READOUTS = [("TokProb", "tokprob"), ("P(IK)", "pik"), ("SAPLMA", "saplma")]
MODEL_LABEL = {
    "internvl3_5_2b": "InternVL3.5-2B",
    "molmo2_4b": "Molmo2-4B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "gemma3_12b": "Gemma-3-12B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--lengths-store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lengths = json.load(open(args.lengths_store))
    models = []
    resister_floor_vals = []
    for m in MODEL_ORDER:
        res = json.load(open(os.path.join(args.store_dir, m, "result3.json")))
        pos = json.load(open(os.path.join(args.store_dir, m, "position_auroc.json")))
        ro = res["readouts"]
        regime = "resist" if ro["pik"]["attacked_auroc"] >= 0.5 and ro["saplma"]["attacked_auroc"] >= 0.5 else "collapse"
        readouts = []
        for label, key in READOUTS:
            readouts.append({
                "label": label,
                "clean": ro[key]["clean_auroc_subset"],
                "attacked": ro[key]["attacked_auroc"],
            })
        if regime == "resist":
            resister_floor_vals.append(ro["pik"]["attacked_auroc"])
            resister_floor_vals.append(ro["saplma"]["attacked_auroc"])
        ap_block = res["answer_preservation"]
        models.append({
            "model": m,
            "label": MODEL_LABEL[m],
            "regime": regime,
            "readouts": readouts,
            "final_locus": pos["by_position"]["final"],
            "cot_median": lengths[m]["cot"]["median"],
            "native_median": lengths[m]["neutral"]["median"],
            "native_long": lengths[m]["native_long"],
            "feasible_rate": ap_block["median_step_feasible_rate"],
            "rejects": ap_block["mean_fresh_rejects_per_item_dir"],
        })
    footnote = {
        "molmo_native_median": lengths["molmo2_4b"]["neutral"]["median"],
        "molmo_frac_ge20": lengths["molmo2_4b"]["neutral"]["frac_ge_20"],
        "resister_floor": min(resister_floor_vals),
    }
    out = {"models": models, "footnote": footnote}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"long_answer_full: {len(models)} models, resister floor {footnote['resister_floor']:.3f}")


if __name__ == "__main__":
    main()
