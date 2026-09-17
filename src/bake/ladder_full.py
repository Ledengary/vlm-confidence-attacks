import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.store))
    rows = []
    for t in d["triples"]:
        rows.append({
            "base": t["base"],
            "verifier": t["verifier"],
            "dataset": t["dataset"],
            "spsa_n": t["spsa_n"],
            "clean": t["c0_transfer"]["C0"]["auroc"],
            "clean_ci": t["c0_transfer"]["C0"]["ci"],
            "transfer": t["c0_transfer"]["transfer"]["auroc"],
            "transfer_ci": t["c0_transfer"]["transfer"]["ci"],
            "hsj": t["hsj"]["mean"],
            "spsa": t["spsa"]["mean"],
            "c2": t["c2"]["mean"],
            "parity_max_abs": t["c2_parity_max_abs"],
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": rows}, open(args.out, "w"), indent=2)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
