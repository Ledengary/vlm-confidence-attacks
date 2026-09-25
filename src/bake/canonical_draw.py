import argparse
import glob
import json
import os
import random

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
REUSE_MODEL = "internvl3_5_2b"
SEED = 23
N_PER = 500


def regen(pipeline, draws, model):
    labels = {}
    with open(os.path.join(pipeline, model, "judged.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                labels[str(r["item_id"])] = int(r["correct"])
    answers = {}
    for s in SET_ORDER:
        for path in glob.glob(os.path.join(pipeline, model, s, "shard_*.jsonl")):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        answers[str(r["item_id"])] = r["answer"]
    out = {}
    for s in SET_ORDER:
        elig = [i for i in answers if i in labels and i in draws[s]]
        for group, want in (("correct", 1), ("wrong", 0)):
            ids = sorted(i for i in elig if labels[i] == want)
            random.Random(SEED).shuffle(ids)
            out[(s, group)] = set(ids[:N_PER])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    draws = {}
    for s in SET_ORDER:
        ids = set()
        with open(os.path.join(args.draws, f"{s}.jsonl")) as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["item_id"])
        draws[s] = ids
    out = {"cells": {}}
    for m in MODEL_ORDER:
        with open(os.path.join(args.pools, m, "pool.json")) as f:
            pj = json.load(f)
        target = int(pj.get("n_per", N_PER)) * 2
        committed = {}
        for p in pj["pool"]:
            committed.setdefault((p["dataset"], p["group"]), set()).add(p["item_id"])
        rg = None if m == REUSE_MODEL else regen(args.pipeline, draws, m)
        out["cells"][m] = {}
        for s in SET_ORDER:
            nc = len(committed.get((s, "correct"), set())) + len(committed.get((s, "wrong"), set()))
            if rg is not None:
                ok = all(rg.get((s, g), set()) == committed.get((s, g), set()) for g in ("correct", "wrong"))
                mark = "yes" if ok else "NO"
            else:
                mark = "reuse"
            out["cells"][m][s] = {"match": mark, "pool_size": nc, "target": target}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
