import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

SOURCES = ["gemma3_12b", "llava_onevision_7b"]
TARGETS = ["internvl3_5_2b", "gemma3_12b", "molmo2_4b", "llava_onevision_7b"]
SETS = ["gqa_val_eval", "vqav2_ood", "pope_ood"]


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


def load_by_id(pattern):
    d = {}
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                d[r["item_id"]] = r
    return d


def pool_ids(channels, model):
    ids = set()
    for f in glob.glob(os.path.join(channels, model, "per_item_s*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    ids.add(json.loads(line)["item_id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawtransfer", required=True)
    ap.add_argument("--channels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    readout = os.path.join(args.rawtransfer, "readout")
    mat = {}
    for src in SOURCES:
        for tgt in TARGETS:
            cl = load_by_id(os.path.join(readout, tgt, "clean_s*.jsonl"))
            ad = load_by_id(os.path.join(readout, tgt, f"{src}_s*.jsonl"))
            for s in SETS:
                ids = [i for i in cl if i in ad and cl[i]["dataset"] == s]
                y = [cl[i]["label"] for i in ids]
                mat[(src, tgt, s)] = auc(y, [cl[i]["clean_tok"] for i in ids]) - auc(
                    y, [ad[i]["adv_tok"] for i in ids]
                )

    biggest = max(((k, v) for k, v in mat.items() if k[0] != k[1]), key=lambda kv: kv[1])

    cells = {}
    for src in SOURCES:
        meta = load_by_id(os.path.join(args.rawtransfer, src, "meta_s*.jsonl"))
        pool = pool_ids(args.channels, src)
        cells[src] = {}
        for s in SETS:
            it = [r for r in meta.values() if r["dataset"] == s and r["item_id"] in pool]
            y = [r["label"] for r in it]
            wb = auc(y, [r["src_clean_tok"] for r in it]) - auc(y, [r["src_adv_tok"] for r in it])
            offs = [(tgt, mat[(src, tgt, s)]) for tgt in TARGETS if tgt != src]
            vals = [v for _, v in offs]
            worst = max(offs, key=lambda tv: tv[1])
            cells[src][s] = {
                "white_box": wb,
                "transfer_mean": float(np.mean(vals)),
                "transfer_max": max(vals),
                "worst_target": worst[0],
            }

    n_offdiag = len(SOURCES) * len(SETS) * (len(TARGETS) - 1)
    out = {
        "sources": SOURCES,
        "cells": cells,
        "n_offdiag": n_offdiag,
        "max_pair": {"source": biggest[0][0], "target": biggest[0][1], "set": biggest[0][2], "value": biggest[1]},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out, "max_pair", biggest)


if __name__ == "__main__":
    main()
