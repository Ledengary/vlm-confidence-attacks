from __future__ import annotations
import argparse
import contextlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.config import DATA_DIR
from src.attacks import gate_v12
from src.attacks.strict_bank import StrictCanonicalBank
OUT = DATA_DIR / 'snowglobe' / 'verifier_escape'
IMG = OUT / 'images'
OBJ = 'saplma'
CONFIG = 'B_corrected_30x2'

@contextlib.contextmanager
def _capture_winner(sink: dict):
    orig = gate_v12.validate_candidate

    def wrapped(adv_u8, clean_u8, eps, pp, official_pv_fn, tol: float=0.0):
        out = orig(adv_u8, clean_u8, eps, pp, official_pv_fn, tol)
        sink['adv_u8'] = adv_u8.detach().cpu().numpy().astype(np.uint8)
        sink['clean_u8'] = clean_u8.detach().cpu().numpy().astype(np.uint8)
        return out
    gate_v12.validate_candidate = wrapped
    try:
        yield
    finally:
        gate_v12.validate_candidate = orig

def _save_png(arr_chw: np.ndarray, path: Path):
    Image.fromarray(np.transpose(arr_chw, (1, 2, 0)), mode='RGB').save(path)

def run(model_key: str, setname: str, gpu: int, item_ids, out_dir: Path, shard: int=0, num_shards: int=1, img_dir: Path=None) -> dict:
    from src.data_prep.canonical import load_manifest
    from src.inference.engine import build_item, load_adapter
    from src.attacks.official_pv import official_pv_fn_factory
    from src.attacks.store import ShardWriter, scan
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    IMG_D = Path(img_dir) if img_dir is not None else IMG
    IMG_D.mkdir(parents=True, exist_ok=True)
    dev = f'cuda:{gpu}'
    adapter, pp, _ = load_adapter(model_key, gpu)
    bank = StrictCanonicalBank(model_key, adapter, dev)
    manifest = {r['item_id']: r for r in load_manifest(model_key)}
    jobs = [(i, d) for i in item_ids for d in ('min', 'max')][shard::num_shards]
    res = scan(sorted(out_dir.glob(f'{model_key}_s*.jsonl')))
    todo = [(i, d) for i, d in jobs if (model_key, i, OBJ, d, CONFIG) not in res.done]
    print(f'[regen {model_key}/{setname} s{shard}] jobs={len(jobs)} todo={len(todo)}', flush=True)
    n_ok = n_err = 0
    t0 = time.time()
    with ShardWriter(out_dir / f'{model_key}_s{shard}.jsonl') as w:
        for n, (iid, direction) in enumerate(todo):
            ti = time.time()
            try:
                row = manifest[iid]
                ctx = build_item(adapter, pp, row, dev)
                off = official_pv_fn_factory(adapter, row)
                sink = {}
                with _capture_winner(sink):
                    rec = gate_v12.run_one(adapter, bank, pp, ctx, iid, model_key, OBJ, direction, CONFIG, off)
                adv_p = IMG_D / f'{iid}__{direction}__adv.png'
                cln_p = IMG_D / f'{iid}__clean.png'
                _save_png(sink['adv_u8'], adv_p)
                if not cln_p.exists():
                    _save_png(sink['clean_u8'], cln_p)
                gl = int(np.abs(sink['adv_u8'].astype(np.int32) - sink['clean_u8'].astype(np.int32)).max())
                rec['adv_png'] = str(adv_p.relative_to(DATA_DIR))
                rec['clean_png_saved'] = str(cln_p.relative_to(DATA_DIR))
                rec['linf_greylevels_saved'] = gl
                rec['label'] = row['label']
                rec['group'] = row['group']
                rec['seconds'] = round(time.time() - ti, 3)
                w.write(rec)
                n_ok += 1
            except Exception as e:
                import traceback
                w.write({'model': model_key, 'item_id': iid, 'objective': OBJ, 'direction': direction, 'config': CONFIG, 'ERROR': str(e)[:300], 'trace': traceback.format_exc()[-500:]})
                n_err += 1
            if (n + 1) % 25 == 0:
                el = time.time() - t0
                print(f'  [regen s{shard}] {n + 1}/{len(todo)} {el / (n + 1):.1f}s/rec eta={(len(todo) - n - 1) * el / (n + 1) / 60:.1f}min', flush=True)
    return {'written_ok': n_ok, 'errors': n_err, 'gpu_seconds': round(time.time() - t0, 1)}
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='llava_onevision_7b')
    ap.add_argument('--set', dest='setname', default='vqav2_ood')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', default=str(OUT / 'base_attack'))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--img-dir', default=None)
    a = ap.parse_args()
    ids = json.load(open(a.items))['item_ids']
    r = run(a.model, a.setname, a.gpu, ids, Path(a.out), a.shard, a.num_shards, img_dir=Path(a.img_dir) if a.img_dir else None)
    print(json.dumps(r, indent=2))
