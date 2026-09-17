from __future__ import annotations
import argparse
import glob
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import DATA_DIR
from src.data_prep.datasets.base import DatasetItem
from src.data_prep.datasets.registry import get_adapter
from src.judge.gpt_vision import GPTVision
_ARTICLES = {'a', 'an', 'the'}

def normalize_answer(s: str) -> str:
    s = (s or '').lower().strip()
    s = re.sub('[^\\w\\s]', ' ', s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return ' '.join(toks)

def exact_match(pred: str, gold: str) -> bool:
    p, g = (normalize_answer(pred), normalize_answer(gold))
    if not g:
        return False
    if p == g:
        return True
    return bool(re.search(f'\\b{re.escape(g)}\\b', p))
PIPE = DATA_DIR / 'pipeline'
DRAW = DATA_DIR / 'draws'
DRAW_OF = {'gqa_train_probe_train': 'gqa_train_probe', 'gqa_train_probe_val': 'gqa_train_probe', 'gqa_val_eval': 'gqa_val_eval', 'vqav2_ood': 'vqav2_ood', 'pope_ood': 'pope_ood'}

def _draws():
    d = {}
    for name in set(DRAW_OF.values()):
        p = DRAW / f'{name}.jsonl'
        if p.exists():
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    d[r['item_id']] = r
    return d

def run(model, workers=100):
    out_path = PIPE / model / 'judged.jsonl'
    done = set()
    if out_path.exists():
        for l in open(out_path):
            l = l.strip()
            if l:
                try:
                    done.add(json.loads(l)['item_id'])
                except Exception:
                    pass
    items = []
    for f in glob.glob(str(PIPE / model / '*' / 'shard_*.jsonl')):
        for l in open(f):
            l = l.strip()
            if l:
                r = json.loads(l)
                if r['item_id'] not in done:
                    items.append((r['item_id'], r['dataset'], r['set'], r['question'], r['gold'], r['answer']))
    print(f'{model}: {len(items)} to judge ({len(done)} done)', flush=True)
    if not items:
        return
    draws = _draws()
    adapters = {ds: get_adapter(ds) for ds in {it[1] for it in items}}
    gpt = GPTVision()
    lock = threading.Lock()
    fh = open(out_path, 'a')
    n = [0]

    def work(item):
        iid, ds, setname, q, gold, pred = item
        try:
            img = adapters[ds].get_image(DatasetItem.from_dict(draws[iid]))
            v = gpt.judge_answer(img, q, gold, pred)
        except Exception:
            v = None
        judged = v is not None
        sm = int(exact_match(pred, gold))
        correct = bool(v) if judged else bool(sm)
        with lock:
            fh.write(json.dumps({'item_id': iid, 'set': setname, 'dataset': ds, 'correct': int(correct), 'strmatch': sm, 'judged': judged}) + '\n')
            fh.flush()
            n[0] += 1
            if n[0] % 1000 == 0:
                print(f'  {model}: {n[0]} judged', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in as_completed([ex.submit(work, it) for it in items]):
            pass
    fh.close()
    print(json.dumps({'model': model, 'n_new': n[0], 'api_calls': gpt.n_calls}), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--workers', type=int, default=100)
    a = ap.parse_args()
    run(a.model, a.workers)
