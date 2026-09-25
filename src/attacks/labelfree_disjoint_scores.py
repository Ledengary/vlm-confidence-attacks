from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
import traceback
from src.config import DATA_DIR, REPO_ROOT
from src.inference.engine import load_adapter, build_item
from src.estimators.objectives import forward_all
from src.attacks.canonical_bank import CanonicalBank
from src.estimators.channels import answer_spec
from src.hidden_states.common import _ds
from src.data_prep.datasets.registry import get_adapter
from src.data_prep.datasets.base import DatasetItem

OUTROOT = str(DATA_DIR / "labelfree_disjoint")
IMGROOT = str(DATA_DIR / "labelfree_disjoint_imgs")
POOLROOT = str(REPO_ROOT / "artifacts" / "pools")
WANT = ("full", "coupled", "ccpsd", "ccps")
KCOMP = 16


def pool_ids(model, dataset):
    ids = set()
    with open(os.path.join(POOLROOT, f"{model}__pool.json")) as f:
        pool = json.load(f)["pool"]
    for r in pool:
        if r.get("dataset") == dataset:
            ids.add(r["item_id"])
    return ids


def draws(dataset):
    d = {}
    with open(os.path.join(str(DATA_DIR), "draws", f"{dataset}.jsonl")) as f:
        for ln in f:
            r = json.loads(ln)
            d[r["item_id"]] = r
    return d


def disjoint_rows(model, dataset):
    pool = pool_ids(model, dataset)
    dr = draws(dataset)
    rows = []
    with open(os.path.join(str(DATA_DIR), "pipeline", model, "judged.jsonl")) as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("set") != dataset:
                continue
            iid = r["item_id"]
            if iid in pool or iid not in dr:
                continue
            rows.append({"item_id": iid, "label": int(bool(r["correct"])), "draw": dr[iid]})
    rows.sort(key=lambda x: x["item_id"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    os.makedirs(OUTROOT, exist_ok=True)
    imgdir = os.path.join(IMGROOT, a.model, a.set)
    os.makedirs(imgdir, exist_ok=True)
    outp = os.path.join(OUTROOT, f"{a.model}__{a.set}__shard{a.shard}of{a.nshards}.jsonl")
    done = set()
    for fp in glob.glob(os.path.join(OUTROOT, f"{a.model}__{a.set}__*.jsonl")):
        for ln in open(fp):
            try:
                done.add(json.loads(ln)["item_id"])
            except Exception:
                pass
    rows = disjoint_rows(a.model, a.set)
    rows = rows[a.shard::a.nshards]
    if a.limit:
        rows = rows[:a.limit]
    adapter, pp, _ = load_adapter(a.model, a.gpu)
    bank = CanonicalBank(a.model, adapter, dev)
    dsad = get_adapter(a.set.split("_")[0] if a.set != "gqa_val_eval" else "gqa")
    t_load = time.time()
    t0 = time.time()
    n = 0
    fout = open(outp, "a")
    for r in rows:
        iid = r["item_id"]
        if iid in done:
            continue
        try:
            di = DatasetItem.from_dict(r["draw"])
            spec = answer_spec(a.set, di.answer_type)
            pngp = os.path.join(imgdir, f"{iid}.png")
            img = _ds(dsad.get_image(di).convert("RGB"))
            tmp = pngp + ".tmp"
            img.save(tmp, format="PNG")
            os.replace(tmp, pngp)
            row = {"item_id": iid, "question": di.question,
                   "format_instruction": spec.format_instruction,
                   "max_new_tokens": spec.max_new_tokens, "clean_png": pngp}
            ctx = build_item(adapter, pp, row, dev)
            out0 = forward_all(adapter, ctx, ctx.pv_official, need_grad=False)
            drow = out0.logits[0, ctx.L - 1]
            order = drow.argsort(descending=True).tolist()
            comp = [t for t in order if t != ctx.ans_ids[0]][:KCOMP]
            bank.coupled_meta[iid] = {"item_id": iid, "ans_ids": ctx.ans_ids, "comp_ids": comp}
            sc = bank.score_all(ctx, out0, iid, want=WANT)
            fout.write(json.dumps({"item_id": iid, "label": r["label"],
                                   "clean_scores": {k: (float(v) if v is not None else None) for k, v in sc.items()}}) + "\n")
            fout.flush()
            n += 1
        except Exception as e:
            sys.stderr.write(f"ITEM_FAIL {iid}: {e}\n")
            traceback.print_exc()
            continue
    dt = time.time() - t0
    print(f"[{a.model}/{a.set} shard{a.shard}/{a.nshards}] scored {n} items in {dt:.1f}s "
          f"(load {t0 - t_load:.1f}s), {dt / max(n, 1):.2f}s/item -> out {outp}")


if __name__ == "__main__":
    main()
