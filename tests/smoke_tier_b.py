import argparse
import json
import os
import pathlib
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "weights"
SMOKE_DATA = REPO / "tests" / "smoke_data"
MODEL = "internvl3_5_2b"


def build_data_root():
    root = pathlib.Path(tempfile.mkdtemp(prefix="tier_b_smoke_"))
    for name in ("coupled", "snowglobe", "ccps", "pipeline"):
        os.symlink(WEIGHTS / name, root / name)
    (root / "iclr2027" / "manifest").mkdir(parents=True)
    os.symlink(SMOKE_DATA / "manifest_internvl_gqa8.jsonl", root / "iclr2027" / "manifest" / f"{MODEL}.jsonl")
    os.symlink(SMOKE_DATA / "adv", root / "adv")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = build_data_root()
    import src.config as C
    C.DATA_DIR = root
    from src.inference.engine import build_item, load_adapter
    from src.attacks.canonical_bank import CanonicalBank
    from src.attacks.official_pv import official_pv_fn_factory
    from src.data_prep.canonical import load_manifest
    import src.attacks.gate_v12 as G
    frozen = json.load(open(SMOKE_DATA / "frozen_sha.json"))
    dev = f"cuda:{a.gpu}"
    t0 = time.time()
    adapter, pp, pp_meta = load_adapter(MODEL, a.gpu)
    bank = CanonicalBank(MODEL, adapter, dev)
    t_load = time.time() - t0
    rows = load_manifest(MODEL)
    if a.limit:
        rows = rows[:a.limit]
    t_infer = 0.0
    t_attack = 0.0
    n = 0
    n_sha = 0
    n_readouts = 0
    for row in rows:
        iid = row["item_id"]
        lab = int(row["label"])
        direction = "min" if lab == 1 else "max"
        ti = time.time()
        ctx = build_item(adapter, pp, row, dev)
        t_infer += time.time() - ti
        off = official_pv_fn_factory(adapter, row)
        ta = time.time()
        rec = G.run_one(adapter, bank, pp, ctx, iid, MODEL, "coupled", direction, "C_corrected_60x3", off)
        t_attack += time.time() - ta
        sha = (rec.get("validity") or {}).get("endpoint_u8_sha256")
        tgt = frozen.get(iid, {}).get(direction)
        n += 1
        n_sha += int(bool(sha and tgt and sha == tgt))
        n_readouts += int(len([k for k, v in (rec.get("endpoint_scores") or {}).items() if v is not None]) >= 7)
    print(f"Tier B smoke test: InternVL3.5-2B, GQA, {n} items")
    print(f"stage load (model + estimator bank): {t_load:.1f}s")
    print(f"stage inference (clean build per item): total {t_infer:.1f}s, mean {t_infer / n:.1f}s")
    print(f"stage attack + seven readouts per item: total {t_attack:.1f}s, mean {t_attack / n:.1f}s")
    print(f"seven readouts scored on every item: {n_readouts}/{n}")
    print(f"endpoint sha256 matches frozen store: {n_sha}/{n}")
    ok = n_sha == n and n_readouts == n
    print("SMOKE TEST PASS" if ok else "SMOKE TEST FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
