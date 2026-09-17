import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

READOUT = "coupled"
CELL_MODEL = "molmo2_4b"
CELL_SET = "pope_ood"


def _oracle(cand, y):
    a = np.asarray(cand, float)
    y = np.asarray(y, int)
    return np.where(y == 1, a.min(axis=1), a.max(axis=1)).astype(float)


def _wins(pos, neg):
    d = pos[:, None] - neg[None, :]
    return float((d > 0).sum() + 0.5 * (d == 0).sum())


def _decompose(score, label, answer):
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    answer = np.asarray(answer, object)
    pos = np.where(label == 1)[0]
    neg = np.where(label == 0)[0]
    npair = len(pos) * len(neg)
    tot = _wins(score[pos], score[neg])
    by = defaultdict(lambda: {"p": [], "n": []})
    for i in pos:
        by[answer[i]]["p"].append(i)
    for j in neg:
        by[answer[j]]["n"].append(j)
    wpairs = 0
    within = 0.0
    for _, g in by.items():
        if g["p"] and g["n"]:
            wpairs += len(g["p"]) * len(g["n"])
            within += _wins(score[np.array(g["p"])], score[np.array(g["n"])])
    xpairs = npair - wpairs
    return wpairs / npair, within / wpairs, (tot - within) / xpairs


def _seed23_from_csv(csv_path):
    for r in csv.DictReader(open(csv_path)):
        if r["model"] == CELL_MODEL and r["dataset"] == CELL_SET and r["estimator"] == READOUT:
            return {"w": float(r["w"]), "a_same": float(r["wa_att"]), "a_cross": float(r["xa_att"])}
    raise SystemExit("seed-23 binding row not found")


def _reseed(records_path, answers):
    cand, y, ans = [], [], []
    for line in open(records_path):
        if not line.strip():
            continue
        r = json.loads(line)
        if "ERROR" in r:
            continue
        row = [r["clean_scores"][READOUT]] + [e["endpoint_scores"][READOUT] for e in r["endpoints"].values()]
        cand.append(row)
        y.append(int(r["label"]))
        ans.append(answers.get(r["item_id"]))
    s = _oracle(cand, y)
    w, a_same, a_cross = _decompose(s, y, ans)
    return {"w": w, "a_same": a_same, "a_cross": a_cross}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed24", required=True)
    ap.add_argument("--seed25", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    answers = {}
    for line in open(a.manifest):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["dataset"] == CELL_SET:
            answers[r["item_id"]] = str(r.get("committed_clean_answer"))
    out = {
        "seed23": _seed23_from_csv(a.csv),
        "seed24": _reseed(a.seed24, answers),
        "seed25": _reseed(a.seed25, answers),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("seed24", out["seed24"], "seed25", out["seed25"])


if __name__ == "__main__":
    main()
