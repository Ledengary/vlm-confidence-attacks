import argparse
import glob
import json
import os

import numpy as np

MODELS = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SETS = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
SEED = 23
N_BOOT = 10000
EST = [("TokProb", "tokprob", None), ("P(IK)", "pik", "pik"), ("SAPLMA", "saplma", "saplma"),
       ("CCPS", "ccps", "ccpsd"), ("CCPS-D", "ccpsd", "ccpsd"), ("FULL", "full", "full"),
       ("COUPLED", "coupled", "coupled")]
OBJS = ["full", "pik", "saplma", "ccpsd", "coupled"]


def auroc(score, label):
    s = np.asarray(score, float); y = np.asarray(label, int)
    pos = s[y == 1]; neg = np.sort(s[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lo = np.searchsorted(neg, pos, side="left"); hi = np.searchsorted(neg, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def boot_ci(score, label, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    s = np.asarray(score, float); y = np.asarray(label, int); m = len(y)
    vals = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, m, m)
        vals[b] = auroc(s[idx], y[idx])
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def load_cell(store, m, ds):
    items = {}
    for f in sorted(glob.glob(os.path.join(store, f"{m}_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["dataset"] != ds:
                continue
            it = items.setdefault(r["item_id"], {"label": int(r["label"]), "clean": r["clean_scores"], "rec": {}, "eps": []})
            it["rec"][(r["objective"], r["direction"])] = r["endpoint_scores"]
            it["eps"].append(r["endpoint_scores"])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--floors", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fl = {(c["model"], c["set"]): c["operative"] for c in json.load(open(a.floors))["cells"]}
    rows = []; tok_variants = {}
    for m in MODELS:
        for ds in SETS:
            items = load_cell(a.store, m, ds)
            ids = sorted(items)
            label = np.array([items[i]["label"] for i in ids], int)
            floor = fl[(m, ds)]
            for name, key, obj in EST:
                clean = auroc([items[i]["clean"][key] for i in ids], label)
                wc = []
                for i in ids:
                    cand = [items[i]["clean"][key]] + [e[key] for e in items[i]["eps"]]
                    wc.append(min(cand) if items[i]["label"] == 1 else max(cand))
                wc_union = auroc(wc, label)
                if name == "TokProb":
                    variants = {}
                    for o in OBJS:
                        adv = [items[i]["rec"][(o, "min" if items[i]["label"] == 1 else "max")][key] for i in ids]
                        variants[o] = auroc(adv, label)
                    tok_variants[f"{m}/{ds}"] = variants
                    best_o = min(variants, key=lambda o: variants[o])
                    adv = np.array([items[i]["rec"][(best_o, "min" if items[i]["label"] == 1 else "max")][key] for i in ids], float)
                    single = variants[best_o]; ci = boot_ci(adv, label); note = f"best={best_o}"
                else:
                    adv = np.array([items[i]["rec"][(obj, "min" if items[i]["label"] == 1 else "max")][key] for i in ids], float)
                    single = auroc(adv, label); ci = boot_ci(adv, label); note = ""
                rows.append({"estimator": name, "model": m, "set": ds, "clean": clean, "single": single,
                             "single_ci": ci, "wc_union": wc_union, "diff": wc_union - single, "floor": floor,
                             "below_chance": bool(single < 0.5), "le_floor": bool(single <= floor), "note": note})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": rows, "tok_variants": tok_variants}, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
