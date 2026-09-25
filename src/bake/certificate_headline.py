import argparse
import glob
import json
import os

import numpy as np

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]
SEED = 23
N_BOOT = 10000
NS = 50000
GRID = np.linspace(0.0, 1.0, 201)
FINE = np.linspace(0.0, 1.0, 20001)
CH = [("tokprob", "tokprob", None), ("pik", "pik", "pik"), ("saplma", "saplma", "saplma"),
      ("ccps", "ccps", "ccpsd"), ("ccpsd", "ccpsd", "ccpsd"), ("FULL", "full", "full"),
      ("COUPLED", "coupled", "coupled")]
OBJS = ["full", "pik", "saplma", "ccpsd", "coupled"]


def _G_vec(diff_sorted, vs):
    n = len(diff_sorted)
    hi = np.searchsorted(diff_sorted, vs, side="right")
    lo = np.searchsorted(diff_sorted, vs, side="left")
    return ((n - hi) + 0.5 * (hi - lo)) / n


def gstar_grid(diff_sorted, floor):
    G = _G_vec(diff_sorted, GRID * 2.0)
    m = G <= floor
    return float(GRID[np.argmax(m)]) if m.any() else float("inf")


def gstar_exact(diff_sorted, floor):
    vs = FINE * 2.0
    G = _G_vec(diff_sorted, vs)
    m = G <= floor
    return float(vs[np.argmax(m)] / 2.0) if m.any() else float("inf")


def gamma_ci(pos, neg, floor):
    rng = np.random.default_rng(SEED)
    gs = np.empty(N_BOOT)
    k = int(np.clip(round((1.0 - floor) * NS), 0, NS - 1))
    for b in range(N_BOOT):
        ic = rng.integers(0, len(pos), NS)
        iw = rng.integers(0, len(neg), NS)
        g = pos[ic] - neg[iw]
        gs[b] = np.partition(g, k)[k] / 2.0
    return [float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))]


def load_cell(store, m, ds):
    items = {}
    for f in sorted(glob.glob(os.path.join(store, f"{m}_s*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["dataset"] != ds:
                continue
            it = items.setdefault(r["item_id"], {"label": int(r["label"]), "clean": r["clean_scores"], "rec": {}})
            it["rec"][(r["objective"], r["direction"])] = (r["endpoint_scores"], bool(r.get("used_clean_fallback")))
    return items


def harmful(clean, endpoint, label):
    return (clean - endpoint) if label == 1 else (endpoint - clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--floors", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fl = {}
    for c in json.load(open(a.floors))["cells"]:
        fl[(c["model"], c["set"])] = c["operative"]
    cells = {(m, ds): load_cell(a.store, m, ds) for m in MODEL_ORDER for ds in SET_ORDER}
    rows = []
    for chname, key, obj in CH:
        for m in MODEL_ORDER:
            for ds in SET_ORDER:
                items = cells[(m, ds)]
                ids = sorted(items)
                label = np.array([items[i]["label"] for i in ids], int)
                clean = np.array([items[i]["clean"][key] for i in ids], float)
                floor = fl[(m, ds)]
                pos = clean[label == 1]; neg = clean[label == 0]
                dd = (pos[:, None] - neg[None, :]).ravel()
                diff = np.sort(dd)
                sep = float(dd.mean())
                g0 = float(_G_vec(diff, np.array([0.0]))[0])
                gg = gstar_grid(diff, floor)
                ge = gstar_exact(diff, floor)
                na = (gg == 0.0)
                ci = [float("nan"), float("nan")] if na else gamma_ci(pos, neg, floor)
                d = []; feas = []
                for i in ids:
                    it = items[i]; lab = it["label"]
                    if chname == "tokprob":
                        best = None; anyf = False
                        for o in OBJS:
                            dirn = "min" if lab == 1 else "max"
                            es, fb = it["rec"][(o, dirn)]
                            if fb:
                                continue
                            anyf = True
                            hv = harmful(it["clean"]["tokprob"], es["tokprob"], lab)
                            best = hv if best is None else max(best, hv)
                        feas.append(anyf); d.append(best if anyf else np.nan)
                    else:
                        dirn = "min" if lab == 1 else "max"
                        es, fb = it["rec"][(obj, dirn)]
                        feas.append(not fb)
                        d.append(abs(es[key] - it["clean"][key]) if not fb else np.nan)
                d = np.array(d, float); feas = np.array(feas, bool)
                df = d[feas]; nfeas = int(feas.sum())
                disp = float(np.mean(df)) if nfeas else float("nan")
                wmax = float(np.max(df)) if nfeas else float("nan")
                if na:
                    frac = float("nan"); refuted = "n/a"
                else:
                    frac = float(np.mean(df > ge)) if nfeas else float("nan")
                    refuted = "yes" if (nfeas and wmax > ge) else "no"
                rows.append({"channel": chname, "model": m, "set": ds, "floor": floor,
                             "gamma_grid": gg, "gamma_exact": ge, "ci": ci, "sep": sep, "g0": g0,
                             "disp": disp, "wmax": wmax, "nfeas": nfeas, "frac": frac, "refuted": refuted})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
