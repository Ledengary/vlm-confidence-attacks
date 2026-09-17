import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

PURIFIERS = [("jpeg75", "JPEG q75"), ("jpeg50", "JPEG q50"), ("blur", "blur"),
             ("bit4", "4-bit"), ("resizecrop", "resize-crop")]
PURIF_MODELS = ["internvl3_5_2b", "gemma3_12b", "llava_onevision_7b"]
SETS = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
NICE = {"internvl3_5_2b": "InternVL", "llava_onevision_7b": "LLaVA"}
DS = {"gqa_val_eval": "GQA", "pope_ood": "POPE"}
SMOOTH_ORDER = [("llava_onevision_7b", "gqa_val_eval", "0.08"),
                ("llava_onevision_7b", "gqa_val_eval", "0.16"),
                ("llava_onevision_7b", "pope_ood", "0.08"),
                ("llava_onevision_7b", "pope_ood", "0.16"),
                ("internvl3_5_2b", "gqa_val_eval", "0.16")]
BASE_MODEL = "llava_onevision_7b"
BASE_SET = "gqa_val_eval"
NOT_RUN = ["molmo2_4b", "gemma3_12b"]


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    p = int((y == 1).sum())
    n = int((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def eot_aim_adv(smoothing_dir, model, dataset, sig):
    recs = {}
    pat = os.path.join(smoothing_dir, model, "eot", f"{dataset}_eot_sig{sig}_*.jsonl")
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            recs[r["item_id"]] = r
    ids = [i for i in recs if recs[i].get("byte_identical")]
    y = [recs[i]["label"] for i in ids]
    s = [recs[i]["adv_score"] for i in ids]
    return float(auc(y, s)), len(ids)


def build_block_a(preprocess_dir):
    defenses = []
    for dk, dlab in PURIFIERS:
        surv = []
        corr = []
        for m in PURIF_MODELS:
            dd = json.load(open(os.path.join(preprocess_dir, m, "preprocess.json")))["datasets"]
            for s in SETS:
                if s in dd and dk in dd[s]:
                    rs = dd[s][dk]["range_survival_vs_nodefense"].get("pik")
                    if isinstance(rs, (int, float)):
                        surv.append(rs)
                    corr.append(dd[s][dk]["answer_corrupted_rate"])
        defenses.append({
            "label": dlab,
            "surv_min": min(surv), "surv_max": max(surv), "surv_med": float(np.median(surv)),
            "corr_min": min(corr), "corr_max": max(corr), "corr_med": float(np.median(corr)),
        })
    return {"defenses": defenses}


def build_block_b(analysis_path, gamma_path, smoothing_dir, gth_path, base_aim_disp):
    an = json.load(open(analysis_path))["cells"]
    sg = json.load(open(gamma_path))
    gth = json.load(open(gth_path))["cells"]
    base_c = gth[f"{BASE_MODEL}/{BASE_SET}"]["saplma"]
    base = {
        "model": BASE_MODEL, "set": BASE_SET,
        "clean": base_c["clean_auroc"], "aim_disp": base_aim_disp,
        "aim_adv": base_c["aimed_adv_auroc"], "floor": base_c["floor"],
        "gamma_star": base_c["gamma_star"],
    }
    base["holds"] = base["aim_adv"] < base["floor"]
    rows = []
    for m, s, sig in SMOOTH_ORDER:
        cell = an[f"{m}/{s}/n32"]
        node = cell["sigmas"][sig]["saplma"]
        g = sg[f"{NICE[m]}/{DS[s]}/sig{sig}"]
        aim_disp = round(g["mean_disp"], 3)
        aim_adv, _ = eot_aim_adv(smoothing_dir, m, s, sig)
        rows.append({
            "model": m, "set": s, "sigma": sig,
            "clean": node["clean_auroc"], "tr_disp": node["delta_amplitude"],
            "aim_disp": aim_disp, "tr_adv": node["adv_auroc"], "aim_adv": aim_adv,
            "floor": cell["floor"], "gamma_star": g["gamma_star"],
            "holds": aim_adv < cell["floor"],
        })
    return {"base": base, "rows": rows, "not_run": NOT_RUN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess-dir", required=True)
    ap.add_argument("--smoothing-analysis", required=True)
    ap.add_argument("--smoothing-gamma", required=True)
    ap.add_argument("--smoothing-dir", required=True)
    ap.add_argument("--gamma-threshold", required=True)
    ap.add_argument("--base-aim-disp", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {
        "block_a": build_block_a(args.preprocess_dir),
        "block_b": build_block_b(args.smoothing_analysis, args.smoothing_gamma, args.smoothing_dir,
                                 args.gamma_threshold, args.base_aim_disp),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} with {len(out['block_a']['defenses'])} purifiers and "
          f"{len(out['block_b']['rows'])} smoothing rows")


if __name__ == "__main__":
    main()
