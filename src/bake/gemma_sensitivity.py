import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
MODEL_TAG = {
    "internvl3_5_2b": "internvl",
    "molmo2_4b": "molmo",
    "llava_onevision_7b": "llava",
    "gemma3_12b": "gemma",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"models": {}}
    for m in MODEL_ORDER:
        path = os.path.join(args.store, "sensitivity", MODEL_TAG[m] + ".json")
        with open(path) as f:
            a = json.load(f)["agg"]
        out["models"][m] = {
            "dither_kl": a["check1_kl"]["dither"],
            "gauss_kl": a["check1_kl"]["gaussian"],
            "nd_rms": a["check2_normspace_dither"]["rms"],
            "clean_entropy": a["check3_position"]["clean_entropy"],
            "logit_gap": a["check4_logit"]["top1_top2_gap"],
        }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("baked gemma_sensitivity " + str(len(MODEL_ORDER)) + " models")


if __name__ == "__main__":
    main()
