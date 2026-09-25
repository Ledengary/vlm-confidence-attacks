import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
MODEL_TAG = {
    "internvl3_5_2b": "internvl",
    "molmo2_4b": "molmo",
    "llava_onevision_7b": "llava",
    "gemma3_12b": "gemma",
}
SET_TAG = {"gqa_val_eval": "gqa", "vqav2_ood": "vqa", "pope_ood": "pope"}
METRICS = ["t1", "t5", "t10", "tv", "stv", "klm"]


def benign_mean(store, tag):
    vals = []
    with open(os.path.join(store, tag, "behavioral_s0.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "ERROR" in rec:
                continue
            if "benign_kl_mean" in rec:
                vals.append(rec["benign_kl_mean"])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.store, "behavioral_agg.json")) as f:
        agg = json.load(f)
    deltas = sorted({d for tag in agg for d in agg[tag]}, key=lambda x: -float(x))
    numer = max(float(d) for d in deltas)
    out = {"deltas": deltas, "models": {}}
    for m in MODEL_ORDER:
        a = agg[MODEL_TAG[m]]
        entry = {}
        for d in deltas:
            entry[d] = {k: a[d][k] for k in METRICS}
        vals = []
        for s in SET_ORDER:
            vals.extend(benign_mean(args.store, MODEL_TAG[m] + "_" + SET_TAG[s]))
        bmean = sum(vals) / len(vals)
        entry["benign_kl_mean"] = bmean
        entry["ratio"] = numer / bmean
        out["models"][m] = entry
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("baked kl_behavioral " + str(len(MODEL_ORDER)) + " models")


if __name__ == "__main__":
    main()
