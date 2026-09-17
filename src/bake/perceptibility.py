import argparse
import json
import os

import numpy as np

MODEL_MAP = {
    "internvl": "internvl3_5_2b",
    "gemma": "gemma3_12b",
    "molmo": "molmo2_4b",
    "llava": "llava_onevision_7b",
}
SET_MAP = {"gqa": "gqa_val_eval", "vqa": "vqav2_ood", "pope": "pope_ood"}
MODEL_TAGS = ["internvl", "gemma", "molmo", "llava"]
SET_TAGS = ["gqa", "vqa", "pope"]


def cell_stats(path):
    recs = []
    with open(path) as f:
        for line in f:
            if not line.strip() or "ERROR" in line:
                continue
            recs.append(json.loads(line))
    moved = [r for r in recs if r.get("moved") and "adv_lpips_aimed" in r]
    if not moved:
        return None
    al = np.array([r["adv_lpips_aimed"] for r in moved])
    ss = np.array([r["adv_ssim_aimed"] for r in moved])
    return {
        "lpips_mean": float(al.mean()),
        "lpips_max": float(al.max()),
        "ssim_mean": float(ss.mean()),
        "ssim_min": float(ss.min()),
        "jpeg_lpips": float(np.mean([r["benign_jpeg95_lpips"] for r in recs])),
        "jpeg_ssim": float(np.mean([r["benign_jpeg95_ssim"] for r in recs])),
        "crop_lpips": float(np.mean([r["benign_crop2_lpips"] for r in recs])),
        "crop_ssim": float(np.mean([r["benign_crop2_ssim"] for r in recs])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"cells": {}}
    jl, js, cl, cs = [], [], [], []
    for mt in MODEL_TAGS:
        for st in SET_TAGS:
            path = os.path.join(args.store, f"{mt}_{st}", "records.jsonl")
            if not os.path.exists(path):
                continue
            v = cell_stats(path)
            if v is None:
                continue
            out["cells"].setdefault(MODEL_MAP[mt], {})[SET_MAP[st]] = {
                "lpips_mean": v["lpips_mean"],
                "lpips_max": v["lpips_max"],
                "ssim_mean": v["ssim_mean"],
                "ssim_min": v["ssim_min"],
            }
            jl.append(v["jpeg_lpips"])
            js.append(v["jpeg_ssim"])
            cl.append(v["crop_lpips"])
            cs.append(v["crop_ssim"])
    out["benign"] = {
        "jpeg_lpips": float(np.mean(jl)),
        "jpeg_ssim": float(np.mean(js)),
        "crop_lpips": float(np.mean(cl)),
        "crop_ssim": float(np.mean(cs)),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"cells {sum(len(v) for v in out['cells'].values())}")


if __name__ == "__main__":
    main()
