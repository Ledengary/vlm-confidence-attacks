from __future__ import annotations
import argparse
import json
import time
import torch
from src.config import DATA_DIR
from src.attacks.attack import AttackConfig, run_direction, validate_candidate
from src.data_prep.canonical import load_manifest
from src.inference.engine import build_item, forward_config, load_adapter, png_bytes_from_raw, sha256_bytes
from src.attacks.official_pv import official_pv_fn_factory
ROOT = DATA_DIR / 'iclr2027'
PRIMARY_CONFIGS = {'A_15x1': AttackConfig(steps=15, n_restarts=1, label='A_15x1'), 'B_30x2': AttackConfig(steps=30, n_restarts=2, label='B_30x2'), 'C_60x3': AttackConfig(steps=60, n_restarts=3, label='C_60x3')}
LEGACY_CONFIG = AttackConfig(steps=15, n_restarts=1, tau_safe=1.0, label='legacy_repaired_15x1_tau1')

def out_dir(run: str, config: str) -> 'object':
    return ROOT / ('primary_' + config if run == 'primary' else 'legacy_repaired')

def _done_ids(path, model=None):
    done = set()
    paths = [path]
    if model is not None:
        paths = sorted(path.parent.glob(f'{model}_s*.jsonl'))
    for p in paths:
        if p.exists():
            for line in open(p):
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)['item_id'])
                    except Exception:
                        pass
    return done

def run(run_kind: str, model: str, gpu: int, shard: int, num_shards: int, config: str, store_png: bool=True):
    cfg = PRIMARY_CONFIGS[config] if run_kind == 'primary' else LEGACY_CONFIG
    d = out_dir(run_kind, config)
    d.mkdir(parents=True, exist_ok=True)
    img_dir = d / 'adv_png' / model
    if store_png:
        img_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(model)
    mine = rows[shard::num_shards]
    out_path = d / f'{model}_s{shard}.jsonl'
    done = _done_ids(out_path, model)
    todo = [r for r in mine if r['item_id'] not in done]
    print(f'[{run_kind} {config} {model} s{shard}/{num_shards} gpu{gpu}] mine={len(mine)} done={len(done)} todo={len(todo)}', flush=True)
    if not todo:
        return
    adapter, pp, pp_meta = load_adapter(model, gpu)
    if shard == 0:
        json.dump({'forward_config': forward_config(adapter, pp_meta), 'attack_config': cfg.to_json(), 'run': run_kind}, open(d / f'{model}_config.json', 'w'), indent=2)
    fh = open(out_path, 'a')
    t0 = time.time()
    n = 0
    for r in todo:
        ti = time.time()
        try:
            ctx = build_item(adapter, pp, r, f'cuda:{gpu}')
            off_fn = official_pv_fn_factory(adapter, r)
            clean_u8 = torch.clamp(torch.round(ctx.raw01 * 255.0), 0, 255).to(torch.uint8).cpu()
            recon_ok = ctx.fresh_answer == r['committed_clean_answer']
            if run_kind == 'primary':
                directions = ('min', 'max')
            else:
                directions = ('min',) if r['label'] == 1 else ('max',)
            res = {}
            for direction in directions:
                td = time.time()
                out = run_direction(adapter, pp, ctx, direction, cfg, official_pv_fn=off_fn)
                adv_u8 = out.pop('adv_u8')
                out['validity'] = validate_candidate(adv_u8, clean_u8, cfg.eps, pp, off_fn)
                out['seconds'] = round(time.time() - td, 3)
                if store_png and (not out['used_clean_fallback']):
                    b = png_bytes_from_raw(adv_u8.to(torch.float32) / 255.0)
                    p = img_dir / f"{r['item_id']}__{direction}.png"
                    with open(p, 'wb') as ph:
                        ph.write(b)
                    out['adv_png'] = str(p.relative_to(DATA_DIR))
                    out['adv_png_sha256'] = sha256_bytes(b)
                res[direction] = out
            rec = {'item_id': r['item_id'], 'model': model, 'dataset': r['dataset'], 'label': r['label'], 'group': r['group'], 'gold': r['gold'], 'run': run_kind, 'config': cfg.label, 'clean_answer': r['committed_clean_answer'], 'clean_recon_byte_identical': recon_ok, 'clean_tokprob': res[directions[0]]['clean_tokprob'], 'committed_clean_tokprob': r['committed_clean_tokprob'], 'ans_ids': ctx.ans_ids, 'gen_ids': ctx.gen_ids, 'n_span_tokens': len(ctx.ans_ids), 'prompt_ids_sha256': sha256_bytes(json.dumps([int(t) for t in ctx.gen_prompt_inputs['input_ids'][0].tolist()]).encode()), 'prompt_len': ctx.L, 'generation_kwargs': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': int(r['max_new_tokens']), 'format_instruction': r['format_instruction']}, 'clean_min_margin': round(float(res[directions[0]]['clean_min_margin']), 4), 'seconds_total': round(time.time() - ti, 3), 'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}
            for direction in directions:
                rec[direction] = res[direction]
            if run_kind == 'primary':
                rec['width'] = round(res['max']['endpoint_tokprob'] - res['min']['endpoint_tokprob'], 6)
            else:
                dd = directions[0]
                rec['legacy_direction'] = dd
                rec['legacy_endpoint'] = res[dd]['endpoint_tokprob']
            fh.write(json.dumps(rec) + '\n')
            n += 1
            if n % 10 == 0:
                fh.flush()
                el = time.time() - t0
                print(f'[{run_kind} {model} s{shard}] {n}/{len(todo)} {el / n:.1f}s/item eta={(len(todo) - n) * el / n / 3600:.2f}h', flush=True)
        except Exception as e:
            import traceback
            fh.write(json.dumps({'item_id': r['item_id'], 'model': model, 'dataset': r['dataset'], 'ERROR': str(e)[:400], 'trace': traceback.format_exc()[-600:]}) + '\n')
            fh.flush()
            print(f"[{run_kind} {model} s{shard}] ERROR {r['item_id']}: {str(e)[:200]}", flush=True)
    fh.close()
    print(f'[{run_kind} {model} s{shard}] DONE n={n} elapsed={time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', choices=['primary', 'legacy'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--config', default='C_60x3', choices=list(PRIMARY_CONFIGS))
    ap.add_argument('--no-png', action='store_true')
    a = ap.parse_args()
    run(a.run, a.model, a.gpu, a.shard, a.num_shards, a.config, store_png=not a.no_png)
