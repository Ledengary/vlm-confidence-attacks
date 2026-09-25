import argparse
import glob
import json
import os

import numpy as np

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
READOUTS = [("TokProb", "tokprob"), ("P(IK)", "pik"), ("SAPLMA", "saplma")]
MODEL_LABEL = {
    "internvl3_5_2b": "InternVL3.5-2B",
    "molmo2_4b": "Molmo2-4B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "gemma3_12b": "Gemma-3-12B",
}
SEED = 23
NBOOT = 10000


def auroc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    p = int((y == 1).sum())
    n = int((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    rk = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[o[j + 1]] == s[o[i]]:
            j += 1
        rk[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((rk[y == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def per_item(store_dir, model, key):
    sub = set(json.load(open(os.path.join(store_dir, model, "attack3_subset.json")))["item_ids"])
    items = {}
    for f in glob.glob(os.path.join(store_dir, model, "attack3", f"{model}_{key}_s*.jsonl")):
        for ln in open(f):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("objective") != key:
                continue
            iid = r["item_id"]
            d = items.setdefault(iid, {"label": int(r["label"])})
            d["clean"] = r["clean_scores"].get(key, r.get("clean_objective"))
            d[r["direction"]] = r["endpoint_scores"].get(key, r.get("endpoint_objective"))
    ids = sorted(i for i in sub if i in items and "min" in items[i] and "max" in items[i])
    y = np.array([items[i]["label"] for i in ids], float)
    clean = np.array([items[i]["clean"] for i in ids], float)
    adv = np.array([min(items[i]["clean"], items[i]["min"], items[i]["max"]) if items[i]["label"] == 1
                    else max(items[i]["clean"], items[i]["min"], items[i]["max"]) for i in ids], float)
    return y, clean, adv


def boot_ci(y, s):
    rng = np.random.default_rng(SEED)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    vals = np.empty(NBOOT)
    for b in range(NBOOT):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        vals[b] = auroc(y[bi], s[bi])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


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
            y, clean_s, adv_s = per_item(args.store_dir, m, key)
            npos, nneg = int((y == 1).sum()), int((y == 0).sum())
            readouts.append({
                "label": label,
                "clean": ro[key]["clean_auroc_subset"],
                "attacked": ro[key]["attacked_auroc"],
                "clean_ci": boot_ci(y, clean_s),
                "attacked_ci": boot_ci(y, adv_s),
                "n": int(len(y)),
                "n_pos": npos,
                "n_neg": nneg,
                "n_pairs": npos * nneg,
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
