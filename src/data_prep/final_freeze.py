from __future__ import annotations
import json
import math
import random
from collections import defaultdict
from typing import Optional
from src.config import DATA_DIR, SEED
from src.data_prep.datasets.registry import get_adapter

def stratified_sample(items, target: Optional[int], seed: int=SEED):
    if target is None or target >= len(items):
        return list(items)
    groups: dict[str, list] = defaultdict(list)
    for it in items:
        groups[it.answer_type].append(it)
    total = len(items)
    raw = {k: target * len(v) / total for k, v in groups.items()}
    alloc = {k: int(math.floor(r)) for k, r in raw.items()}
    remaining = target - sum(alloc.values())
    for k in sorted(groups, key=lambda k: raw[k] - alloc[k], reverse=True)[:remaining]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = []
    for k, v in groups.items():
        n = min(alloc[k], len(v))
        idx = sorted(rng.sample(range(len(v)), n))
        chosen.extend((v[j] for j in idx))
    return chosen
DRAW_DIR = DATA_DIR / 'draws'
RECORDS_DIR = DATA_DIR / 'final_records'
SAFETY = 1.18
FREEZE_SETS = [('gqa_val_eval', 'gqa', 'val', 10000, 2.8, True), ('gqa_train_probe', 'gqa', 'train', 25000, 2.8, True), ('vqav2_ood', 'vqav2', 'validation', 10000, 1.55, False), ('pope_ood', 'pope', 'test', None, None, False)]
REQUIRE_NATIVE = {s[0]: s[5] for s in FREEZE_SETS}

def draw_set(name, dataset, split, target, mult) -> int:
    adapter = get_adapter(dataset)
    items = adapter.load_split(split)
    if target is None:
        order = stratified_sample(items, None, seed=SEED)
    else:
        M = min(len(items), math.ceil(target * mult * SAFETY))
        order = stratified_sample(items, M, seed=SEED)
    rng = random.Random(SEED)
    rng.shuffle(order)
    DRAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DRAW_DIR / f'{name}.jsonl'
    with open(path, 'w') as f:
        for i, it in enumerate(order):
            d = it.to_dict()
            d['draw_index'] = i
            f.write(json.dumps(d) + '\n')
    print(f'{name}: drew {len(order)} raw items (source {len(items)}) -> {path.name}', flush=True)
    return len(order)

def draw_all():
    for name, dataset, split, target, mult, _req in FREEZE_SETS:
        draw_set(name, dataset, split, target, mult)
if __name__ == '__main__':
    draw_all()
