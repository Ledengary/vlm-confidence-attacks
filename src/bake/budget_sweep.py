import argparse
import glob
import json
import os

import numpy as np

MODELS = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SETS = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
MSHORT = {"internvl3_5_2b": "internvl", "molmo2_4b": "molmo", "llava_onevision_7b": "llava", "gemma3_12b": "gemma"}
SSHORT = {"gqa_val_eval": "gqa", "vqav2_ood": "vqa", "pope_ood": "pope"}
RD = [("pik", "pik"), ("saplma", "saplma"), ("FULL", "full")]


def tag(m, s):
    return f"{MSHORT[m]}_{SSHORT[s]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_dir", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--gamma", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    BR = json.load(open(os.path.join(a.budget_dir, "budget_results.json")))
    res = BR["results"]; pbc = BR["profiles_below_chance"]
    GT = json.load(open(a.gamma))["cells"]

    present = [e for e in ("2", "4", "8") if all(str(e) in res[tag(m, s)]["eps"] for m in MODELS for s in SETS)]

    upper = {}
    for m in MODELS:
        for s in SETS:
            c = res[tag(m, s)]["eps"]
            by = {e: {"oracle": c[e]["median_oracle"], "lf": c[e]["median_labelfree"]} for e in present}
            upper[f"{m}/{s}"] = {"clean": c[present[0]]["median_clean"], "by_eps": by}

    def load_records(path):
        recs = {}
        for line in open(path):
            r = json.loads(line); recs[r["item_id"]] = r
        return recs

    def load_store_subset(m, s, ids):
        per = {}
        for f in sorted(glob.glob(os.path.join(a.store, f"{m}_s*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                if r["dataset"] != s or r["item_id"] not in ids:
                    continue
                d = per.setdefault(r["item_id"], {"ep": {}, "fb": {}})
                d["ep"][(r["objective"], r["direction"])] = r["endpoint_scores"]
                d["fb"][(r["objective"], r["direction"])] = bool(r.get("used_clean_fallback"))
        return per

    lower = {}
    fallback = {e: [0, 0] for e in present}
    for m in MODELS:
        agg = {gk: {e: {"w": [], "g": []} for e in present} for gk, _ in RD}
        for s in SETS:
            recs = {}
            for e in present:
                p = os.path.join(a.budget_dir, f"eps{e}", tag(m, s), "records.jsonl")
                recs[e] = load_records(p) if os.path.exists(p) else {}
            ids4 = set(recs.get("4", {})) or set(next(iter(recs.values()), {}))
            store8 = load_store_subset(m, s, ids4) if "8" in present else {}
            for gk, obj in RD:
                gstar = GT[f"{m}/{s}"][gk]["gamma_star_exact"]
                for e in present:
                    if e == "8":
                        for iid in ids4:
                            r8 = store8.get(iid)
                            if not r8:
                                continue
                            w = r8["ep"][(obj, "max")][gk if gk != "FULL" else "full"] - r8["ep"][(obj, "min")][gk if gk != "FULL" else "full"]
                            agg[gk][e]["w"].append(w); agg[gk][e]["g"].append(gstar)
                    else:
                        for iid, r in recs[e].items():
                            sk = gk if gk != "FULL" else "full"
                            w = r["endpoints"][f"{obj}|max"]["endpoint_scores"][sk] - r["endpoints"][f"{obj}|min"]["endpoint_scores"][sk]
                            agg[gk][e]["w"].append(w); agg[gk][e]["g"].append(gstar)
            for e in present:
                if e == "8":
                    for iid in ids4:
                        r8 = store8.get(iid)
                        if r8:
                            for k in r8["ep"]:
                                fallback[e][0] += 1; fallback[e][1] += int(r8["fb"].get(k, False))
                else:
                    for iid, r in recs[e].items():
                        for k, v in r["endpoints"].items():
                            fallback[e][0] += 1; fallback[e][1] += int(bool(v.get("used_clean_fallback")))
        for gk, obj in RD:
            byd = {}
            for e in present:
                w = np.array(agg[gk][e]["w"]); g = np.array(agg[gk][e]["g"])
                byd[e] = {"meanW": float(w.mean()) if len(w) else float("nan"),
                          "frac": float((w > g).mean()) if len(w) else float("nan")}
            lower[f"{m}/{gk}"] = {"by_eps": byd}

    out = {"budgets": present, "upper": upper, "lower": lower,
           "pbc": {e: {"oracle": pbc[e]["oracle_below_0.5"], "labelfree": pbc[e]["labelfree_below_0.5"],
                       "n": pbc[e]["n_profiles"]} for e in present if e in pbc},
           "fallback": {e: (fallback[e][1] / fallback[e][0] if fallback[e][0] else 0.0) for e in present}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out, "budgets", present)


if __name__ == "__main__":
    main()
