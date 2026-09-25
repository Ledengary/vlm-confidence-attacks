import argparse
import glob
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
OBJS = ["full", "pik", "saplma", "ccpsd", "coupled"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"cells": {}}
    for m in MODEL_ORDER:
        counts = {}
        items = {}
        any_fb = {}
        for f in sorted(glob.glob(os.path.join(a.store, f"{m}_s*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                s = r["dataset"]; o = r["objective"]; d = r["direction"]
                counts.setdefault((s, o, d), [0, 0])
                counts[(s, o, d)][0] += 1
                if r.get("used_clean_fallback"):
                    counts[(s, o, d)][1] += 1
                    any_fb.setdefault(s, set()).add(r["item_id"])
                items.setdefault(s, set()).add(r["item_id"])
        out["cells"][m] = {}
        for s in SET_ORDER:
            full_e = counts.get((s, "full", "min"), [0, 0])[0] + counts.get((s, "full", "max"), [0, 0])[0]
            full_fb = counts.get((s, "full", "min"), [0, 0])[1] + counts.get((s, "full", "max"), [0, 0])[1]
            tot = sum(counts.get((s, o, d), [0, 0])[0] for o in OBJS for d in ("min", "max"))
            tot_fb = sum(counts.get((s, o, d), [0, 0])[1] for o in OBJS for d in ("min", "max"))
            out["cells"][m][s] = {
                "n_items": len(items.get(s, set())),
                "full_fb": full_fb, "full_endpoints": full_e,
                "full_fb_pct": 100.0 * full_fb / full_e if full_e else 0.0,
                "all_fb_pct": 100.0 * tot_fb / tot if tot else 0.0,
                "items_any_fb": len(any_fb.get(s, set())),
                "excluded": 0,
            }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
