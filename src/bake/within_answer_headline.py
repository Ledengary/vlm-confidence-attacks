import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
MODEL_LONG = {"internvl3_5_2b": "InternVL3.5-2B", "molmo2_4b": "Molmo2-4B",
              "llava_onevision_7b": "LLaVA-OV-7B", "gemma3_12b": "Gemma-3-12B"}
EST = [("P(IK)", "pik", "pik"), ("SAPLMA", "saplma", "saplma"),
       ("CCPS", "ccpsd", "ccps"), ("CCPS-D", "ccpsd", "ccpsd"),
       ("FULL", "full", "full"), ("COUPLED", "coupled", "coupled")]
DEP = ["P(IK)", "SAPLMA", "CCPS", "CCPS-D"]


def wins_total(score, label):
    s = np.asarray(score, float); y = np.asarray(label, int)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0
    a = pos[:, None]; b = neg[None, :]
    return float((a > b).sum() + 0.5 * (a == b).sum()), int(len(pos) * len(neg))


def within_auroc(score, label, answer):
    score = np.asarray(score, float); label = np.asarray(label, int); answer = np.asarray(answer, object)
    by = defaultdict(list)
    for i in range(len(score)):
        by[answer[i]].append(i)
    ww = 0.0; wp = 0
    for _, idx in by.items():
        idx = np.array(idx); la = label[idx]
        if la.min() == la.max():
            continue
        w_, n_ = wins_total(score[idx], la)
        ww += w_; wp += n_
    return ww / wp if wp else float("nan")


def load_answers(meta, m, ds):
    ans = {}
    for f in sorted(glob.glob(os.path.join(meta, m, f"{ds}_meta_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line); iid = r.get("item_id")
            if iid is not None:
                ans[iid] = str(r.get("clean_answer"))
    return ans


def load_cell(store, m, ds):
    items = {}
    for f in sorted(glob.glob(os.path.join(store, f"{m}_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["dataset"] != ds:
                continue
            it = items.setdefault(r["item_id"], {"label": int(r["label"]), "rec": {}})
            it["rec"][(r["objective"], r["direction"])] = (r["clean_scores"], r["endpoint_scores"])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = {}
    for m in MODEL_ORDER:
        for ds in SET_ORDER:
            ans = load_answers(a.meta, m, ds)
            items = load_cell(a.store, m, ds)
            ids = sorted(items)
            label = np.array([items[i]["label"] for i in ids], int)
            answer = np.array([ans[i] for i in ids], object)
            for lbl, obj, key in EST:
                cl = []; ad = []
                for i in ids:
                    it = items[i]
                    d = "min" if it["label"] == 1 else "max"
                    cs, es = it["rec"][(obj, d)]
                    cl.append(cs[key]); ad.append(es[key])
                cells[(m, ds, lbl)] = {"clean": within_auroc(cl, label, answer),
                                       "adv": within_auroc(ad, label, answer)}
    est_rows = []
    for lbl, obj, key in EST:
        cl = [cells[(m, ds, lbl)]["clean"] for m in MODEL_ORDER for ds in SET_ORDER]
        ad = [cells[(m, ds, lbl)]["adv"] for m in MODEL_ORDER for ds in SET_ORDER]
        est_rows.append({"label": lbl, "clean_mean": statistics.mean(cl), "clean_median": statistics.median(cl),
                         "clean_min": min(cl), "clean_max": max(cl), "adv_mean": statistics.mean(ad),
                         "adv_min": min(ad), "adv_max": max(ad), "dagger": lbl == "COUPLED"})
    per_model = [{"label": MODEL_LONG[m], "auroc": statistics.mean([cells[(m, ds, e)]["adv"] for ds in SET_ORDER for e in DEP])}
                 for m in MODEL_ORDER]
    coupled = {f"{m}/{ds}": cells[(m, ds, "COUPLED")]["adv"] for m in MODEL_ORDER for ds in SET_ORDER}
    cp = {"twelve": statistics.mean(list(coupled.values())), "molmo_pope": coupled["molmo2_4b/pope_ood"],
          "eleven": statistics.mean([v for k, v in coupled.items() if k != "molmo2_4b/pope_ood"])}
    out = {"estimators": est_rows, "per_model": per_model, "coupled": cp}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
