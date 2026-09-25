from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.config import DATA_DIR, SEED
from src.attacks.attack import project, quantized_uint8
from src.estimators.objectives import constraint_from_out, forward_all
from src.verifier_ladder.verifier import CORRECTNESS_PROMPT
OUT = DATA_DIR / 'snowglobe' / 'verifier_escape'
IMG = OUT / 'images'
EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 30

def _gemma_diff_pp(ver_adapter):
    from src.inference.families import _gemma_params, _image_processor
    from src.inference import diffpreproc as dp
    ip = _image_processor(ver_adapter.processor)
    p = _gemma_params(ip)

    def pp(raw01):
        return dp.gemma3_pp(raw01, p['height'], p['width'], p['resample'], p['mean'], p['std'])
    return pp

def _verifier_yes_logit(ver_adapter, gemma_pp, raw01, question, base_answer, yes_ids):
    prompt = f'Question: {question}\nProposed answer: {base_answer}\n{CORRECTNESS_PROMPT}'
    inputs = ver_adapter.build_inputs(Image.new('RGB', (32, 32)), prompt)
    pv = gemma_pp(raw01).to(inputs['pixel_values'].dtype).unsqueeze(0) if gemma_pp(raw01).dim() == 3 else gemma_pp(raw01).to(inputs['pixel_values'].dtype)
    inputs = dict(inputs)
    inputs['pixel_values'] = pv
    out = ver_adapter.model(**inputs, use_cache=False)
    logits = out.logits[0, -1].float()
    return torch.logsumexp(logits[yes_ids], dim=0)

def run(gpu: int, item_ids, gamma: float=1.0, shard: int=0, num_shards: int=1, base_model='llava_onevision_7b', verifier_model='gemma3_12b', setname='vqav2_ood', suffix='') -> dict:
    from src.data_prep.canonical import load_manifest
    from src.inference.engine import build_item, fresh_decode, load_adapter
    from src.attacks.official_pv import official_pv_fn_factory
    from src.attacks.strict_bank import StrictCanonicalBank
    from src.attacks.gate_v12 import objective_logit as base_obj
    from src.verifier_ladder.data import load_cell
    from .verifier import Verifier
    if verifier_model != 'gemma3_12b':
        raise NotImplementedError('white-box C2 differentiable verifier preprocessing is wired for the Gemma verifier only in this pilot; use the SPSA query attack for other verifier families')
    dev = f'cuda:{gpu}'
    base_ad, base_pp, _ = load_adapter(base_model, gpu)
    ver_ad, _, _ = load_adapter(verifier_model, gpu)
    bank = StrictCanonicalBank(base_model, base_ad, dev)
    V = Verifier(ver_ad)
    yes_ids = torch.tensor(V.yes_ids, device=dev)
    gemma_pp = _gemma_diff_pp(ver_ad)
    manifest = {r['item_id']: r for r in load_manifest(base_model)}
    cell = load_cell(base_model, setname)
    lab = {i: int(y) for i, y in zip(cell['ids'], cell['y'])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'c2{suffix}_images').mkdir(exist_ok=True)
    mine = item_ids[shard::num_shards]
    out_path = OUT / f'c2{suffix}_scores_s{shard}.jsonl'
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if line.strip():
                done.add(tuple(json.loads(line)['key']))
    fh = open(out_path, 'a')
    n_ok = n_err = 0
    for iid in mine:
        row = manifest[iid]
        y = lab[iid]
        q, ans = (row['question'], row['committed_clean_answer'])
        try:
            ctx = build_item(base_ad, base_pp, row, dev)
            off = official_pv_fn_factory(base_ad, row)
            x0 = ctx.raw01.detach().to(torch.float32)
            committed = ctx.row['committed_clean_answer']
            clean_u8 = quantized_uint8(x0).cpu().numpy().astype(np.uint8)
            d_ver = -1.0 if y == 1 else 1.0
            d_base = -1.0 if y == 1 else 1.0
            for direction in ('min', 'max'):
                key = (iid, direction)
                if key in done:
                    continue
                x = x0.clone()
                cands = []
                for step in range(STEPS + 1):
                    x = x.detach().requires_grad_(True)
                    out = forward_all(base_ad, ctx, base_pp(x), need_grad=True)
                    base_logit, _ = base_obj(bank, ctx, out, iid, 'saplma')
                    con = constraint_from_out(ctx, out, tau_safe=0.0)
                    vlogit = _verifier_yes_logit(ver_ad, gemma_pp, x, q, ans, yes_ids)
                    obj = d_base * base_logit + gamma * d_ver * vlogit
                    if bool(con['feasible']):
                        harm = float(d_ver * vlogit.detach())
                        cands.append((harm, quantized_uint8(x).detach().cpu().numpy()))
                    if step == STEPS:
                        break
                    g = torch.autograd.grad(obj, x)[0]
                    with torch.no_grad():
                        x = project(x + ALPHA * torch.nan_to_num(g).sign(), x0, EPS).detach()
                cands.sort(key=lambda c: -c[0])
                chosen, preserved = (clean_u8, True)
                for harm, u8 in cands[:8]:
                    text, _ = fresh_decode(base_ad, ctx, off(torch.from_numpy(u8.astype(np.uint8))).to(dev))
                    if text == committed:
                        chosen = u8.astype(np.uint8)
                        break
                else:
                    preserved = True
                is_clean = bool(np.array_equal(chosen, clean_u8))
                p = OUT / f'c2{suffix}_images' / f'{iid}__{direction}__c2.png'
                Image.fromarray(np.transpose(chosen, (1, 2, 0)), 'RGB').save(p)
                gl = int(np.abs(chosen.astype(np.int32) - clean_u8.astype(np.int32)).max())
                vscore = V.score_image_file(p, q, ans)['score']
                fh.write(json.dumps({'key': list(key), 'item_id': iid, 'direction': direction, 'label': y, 'answer_preserved': bool(preserved), 'used_clean_fallback': is_clean, 'n_feasible_candidates': len(cands), 'linf_greylevels': gl, 'verifier_score': vscore, 'c2_png': str(p.relative_to(DATA_DIR)), 'gamma': gamma}) + '\n')
                fh.flush()
                n_ok += 1
        except Exception as e:
            import traceback
            fh.write(json.dumps({'key': [iid, 'err'], 'item_id': iid, 'ERROR': str(e)[:300], 'trace': traceback.format_exc()[-400:]}) + '\n')
            fh.flush()
            n_err += 1
        print(f'[c2 s{shard}] {iid} done ok={n_ok} err={n_err}', flush=True)
    fh.close()
    return {'ok': n_ok, 'err': n_err}

def analyze(suffix='') -> dict:
    import glob
    from src.verifier_ladder.data import auroc
    rows = []
    for f in sorted(glob.glob(str(OUT / f'c2{suffix}_scores_s*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if 'ERROR' not in r:
                    rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r['item_id'], {})[r['direction']] = r
    y, s_oracle, preserved, gls = ([], [], 0, [])
    tot = 0
    for iid, dd in by.items():
        vals = [v['verifier_score'] for v in dd.values()]
        lab = list(dd.values())[0]['label']
        y.append(lab)
        s_oracle.append(min(vals) if lab == 1 else max(vals))
        for v in dd.values():
            tot += 1
            preserved += int(v['answer_preserved'])
            gls.append(v['linf_greylevels'])
    y = np.array(y)
    return {'n_items': len(y), 'n_attacks': tot, 'answer_preserved_rate': round(preserved / max(tot, 1), 4), 'max_linf_greylevels': int(max(gls)) if gls else None, 'C2_verifier_auroc_oracle': round(auroc(y, np.array(s_oracle)), 4)}
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--items', required=True)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--base', default='llava_onevision_7b')
    ap.add_argument('--verifier', default='gemma3_12b')
    ap.add_argument('--set', dest='setname', default='vqav2_ood')
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    if a.analyze:
        print(json.dumps(analyze(a.suffix), indent=2))
    else:
        ids = json.load(open(a.items))['item_ids']
        print(json.dumps(run(a.gpu, ids, a.gamma, a.shard, a.num_shards, a.base, a.verifier, a.setname, a.suffix), indent=2))
