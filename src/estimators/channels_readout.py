from __future__ import annotations
import argparse, glob, hashlib, json, math, random
import numpy as np
import torch
from PIL import Image
from collections import Counter
from src.config import DATA_DIR
from src.estimators.channels import answer_spec
from src.hidden_states.common import _score_inputs, _inject_pv
from src.inference.models.factory import build_adapter
from src.inference.models.base import parse_verbalized_confidence
from src.data_prep.datasets.base import DatasetItem
from src.data_prep.datasets.registry import get_adapter as get_ds_adapter
CH = DATA_DIR / 'adv' / 'channels'
DATASETS = {'gqa_val_eval': 'gqa', 'vqav2_ood': 'vqav2', 'pope_ood': 'pope'}
SEED = 23
N_SC = 40
SC_T = 0.7
SC_TOPK = 40

def _load_meta(model):
    items = {}
    for f in glob.glob(str(CH / model / 'per_item_s*.jsonl')):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                items[r['item_id']] = r
    return items

def _done_ids(model, tag):
    done = set()
    for f in glob.glob(str(CH / model / f'readout_{tag}_s*.jsonl')):
        for l in open(f):
            if l.strip():
                done.add(json.loads(l)['item_id'])
    return done

def _new_ids(seq, n_new, ilen):
    return seq[-n_new:] if n_new > 0 else seq[ilen:]

def _sample_answers(adapter, inputs, delta, n, chunk, spec, gen_seed):
    inp = _inject_pv(adapter, inputs, delta) if delta is not None else dict(inputs)
    ilen = inp['input_ids'].shape[1]
    outs = []
    remaining = n
    while remaining > 0:
        k = min(chunk, remaining)
        torch.manual_seed(gen_seed)
        gen = adapter.model.generate(**inp, max_new_tokens=spec.max_new_tokens, do_sample=True, temperature=SC_T, top_k=SC_TOPK, num_return_sequences=k, output_scores=True, return_dict_in_generate=True, pad_token_id=adapter._pad_id())
        n_new = len(gen.scores) if gen.scores is not None else 0
        for j in range(gen.sequences.shape[0]):
            raw = adapter.tokenizer.decode(_new_ids(gen.sequences[j], n_new, ilen), skip_special_tokens=True)
            outs.append(adapter.parse_answer(raw))
        remaining -= k
    return outs

def _sc_conf(answers):
    norm = [a.strip().lower() for a in answers]
    c = Counter(norm)
    maj, cnt = c.most_common(1)[0]
    return (cnt / len(norm), maj)

def _verbalized(adapter, image, question, answer_text, spec, delta):
    vi = adapter.build_verbalized_inputs(image, question, answer_text, format_instruction=spec.format_instruction)
    if delta is not None:
        vi = _inject_pv(adapter, vi, delta)
    ilen = vi['input_ids'].shape[1]
    gen = adapter.model.generate(**vi, max_new_tokens=24, do_sample=False, num_beams=1, output_scores=True, return_dict_in_generate=True, pad_token_id=adapter._pad_id())
    n_new = len(gen.scores) if gen.scores is not None else 0
    raw = adapter.tokenizer.decode(_new_ids(gen.sequences[0], n_new, ilen), skip_special_tokens=True)
    return (parse_verbalized_confidence(raw), raw)

def run(model, tag, gpu=0, shard=0, num_shards=8, redistribute=False, out_shard=None, n_sc=N_SC, chunk=40):
    d = CH / model
    meta = _load_meta(model)
    verified = [r for r in meta.values() if r.get('byte_identical')]
    verified.sort(key=lambda r: r['item_id'])
    n = len(verified)
    if redistribute:
        gd = _done_ids(model, tag)
        undone = [r for r in verified if r['item_id'] not in gd]
        mine = undone[shard::num_shards]
    else:
        per = -(-n // num_shards)
        lo = shard * per
        hi = min(lo + per, n)
        mine = verified[lo:hi]
    fidx = out_shard if out_shard is not None else shard
    out_jsonl = d / f'readout_{tag}_s{fidx}.jsonl'
    done = set()
    if out_jsonl.exists():
        for l in open(out_jsonl):
            if l.strip():
                done.add(json.loads(l)['item_id'])
    todo = [r for r in mine if r['item_id'] not in done]
    print(f'[readout {tag} {model} s{shard}/{num_shards} gpu{gpu}] verified={n} mine={len(mine)} done={len(done)} todo={len(todo)}', flush=True)
    if not todo:
        return
    dev = f'cuda:{gpu}'
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    adapter = build_adapter(model, device=dev)
    adapter.load()
    for p_ in adapter.model.parameters():
        p_.requires_grad_(False)
    draws = {s: {json.loads(l)['item_id']: json.loads(l) for l in open(DATA_DIR / 'draws' / f'{s}.jsonl') if l.strip()} for s in DATASETS}
    fj = open(out_jsonl, 'a')
    nrec = 0
    for r in todo:
        iid = r['item_id']
        setname = r['dataset']
        dataset = DATASETS[setname]
        row = draws[setname][iid]
        it = DatasetItem.from_dict(row)
        spec = answer_spec(dataset, row.get('answer_type'))
        question = r['question']
        clean_answer = r['clean_answer']
        try:
            img = Image.open(DATA_DIR / r['clean_png']).convert('RGB')
            delta = torch.tensor(np.load(DATA_DIR / r['delta_npy']).astype(np.float32), device=dev)
            gseed = SEED + int(hashlib.md5(iid.encode()).hexdigest(), 16) % 100000
            rec = {'item_id': iid, 'dataset': setname, 'group': r['group'], 'label': r['label'], 'direction': r['direction']}
            if tag == 'verb':
                vc, vc_raw = _verbalized(adapter, img, question, clean_answer, spec, None)
                va, va_raw = _verbalized(adapter, img, question, clean_answer, spec, delta)
                rec['verb_clean'] = vc
                rec['verb_adv'] = va
                rec['verb_raw_clean'] = vc_raw
                rec['verb_raw_adv'] = va_raw
            else:
                inputs = adapter.build_inputs(img, adapter.answer_prompt(question, spec.format_instruction))
                a_clean = _sample_answers(adapter, inputs, None, n_sc, chunk, spec, gseed)
                a_adv = _sample_answers(adapter, inputs, delta, n_sc, chunk, spec, gseed)
                sc_c, maj_c = _sc_conf(a_clean)
                sc_a, maj_a = _sc_conf(a_adv)
                rec.update({'sc_clean': round(sc_c, 5), 'sc_adv': round(sc_a, 5), 'n_sc': n_sc, 'maj_clean': maj_c, 'maj_adv': maj_a, 'maj_clean_is_greedy': maj_c == clean_answer.strip().lower(), 'samples_clean': a_clean, 'samples_adv': a_adv})
            fj.write(json.dumps(rec) + '\n')
            fj.flush()
            nrec += 1
            del delta
            if nrec % 25 == 0:
                torch.cuda.empty_cache()
                print(f'[readout {tag} {model} s{shard}] {nrec}/{len(todo)}', flush=True)
        except Exception as e:
            print(f'err {iid}: {str(e)[:150]}', flush=True)
    fj.close()
    print(json.dumps({'model': model, 'tag': tag, 'shard': shard, 'n_written': nrec}), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True, choices=['verb', 'sc'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=8)
    ap.add_argument('--redistribute', action='store_true')
    ap.add_argument('--out-shard', type=int, default=None)
    ap.add_argument('--n-sc', type=int, default=N_SC)
    ap.add_argument('--chunk', type=int, default=40)
    a = ap.parse_args()
    run(a.model, a.tag, a.gpu, a.shard, a.num_shards, a.redistribute, a.out_shard, a.n_sc, a.chunk)
