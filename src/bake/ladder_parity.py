import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--parity-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.store))
    fam = {}
    molmo_triples = []
    for t in d["triples"]:
        v = t["verifier"]
        fam.setdefault(v, {"max": 0.0, "n": 0})
        fam[v]["max"] = max(fam[v]["max"], t["c2_parity_max_abs"])
        fam[v]["n"] += 1
        if v == "molmo":
            molmo_triples.append(t["triple"])
    vals = []
    for sp in molmo_triples:
        for f in glob.glob(os.path.join(args.parity_dir, f"c2_{sp}_*parity*.jsonl")):
            for line in open(f):
                if line.strip():
                    vals.append(abs(float(json.loads(line)["abs_diff"])))
    out = {
        "families": {v: {"max_gap": fam[v]["max"], "n": fam[v]["n"]} for v in fam},
        "molmo_mean_gap": sum(vals) / len(vals) if vals else 0.0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("families", list(fam), "mean_gap", out["molmo_mean_gap"])


if __name__ == "__main__":
    main()
