import argparse
import json
import os

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
CHANNELS = ["tokprob", "pik", "saplma"]
K_LEVELS = [4, 8, 16]
MODEL_LABEL = {
    "internvl3_5_2b": "InternVL3.5-2B",
    "molmo2_4b": "Molmo2-4B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "gemma3_12b": "Gemma-3-12B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = []
    for m in MODEL_ORDER:
        d = json.load(open(os.path.join(args.store_dir, f"{m}__agg_c5k.json")))
        sap = d["loci"]["sap"]["k"]
        rows = []
        for k in K_LEVELS:
            node = sap[str(k)]
            h2 = node["H2"]
            coupled = sum(h2[c]["energy_coupled_mean"] for c in CHANNELS) / len(CHANNELS) * 100.0
            free = sum(h2[c]["energy_free_mean"] for c in CHANNELS) / len(CHANNELS) * 100.0
            random = sum(h2[c]["energy_coupled_random_mean"] for c in CHANNELS) / len(CHANNELS) * 100.0
            varfrac = node["coupled_var_frac_mean"] * 100.0
            rows.append({"k": k, "coupled": coupled, "free": free, "random": random, "varfrac": varfrac})
        models.append({"model": m, "label": MODEL_LABEL[m], "dim": d["dim"], "rows": rows})
    out = {"models": models}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"answer_coupled_subspace: {len(models)} models x {len(K_LEVELS)} k-levels")


if __name__ == "__main__":
    main()
