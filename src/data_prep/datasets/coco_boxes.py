from __future__ import annotations
import glob
import json
import os
from pathlib import Path
from typing import Optional
from src.config import DATA_DIR
COCO_DIR = DATA_DIR / 'coco'
INDEX_PATH = COCO_DIR / 'coco_boxes_by_image_id.json'
_HUB = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'
_COCO_REPO_DIR = _HUB / 'datasets--detection-datasets--coco'

def _category_names() -> list[str]:
    from datasets import load_dataset_builder
    info = load_dataset_builder('detection-datasets/coco').info
    objfeat = info.features['objects']
    sub = objfeat.feature if hasattr(objfeat, 'feature') else objfeat
    node = sub['category']
    for _ in range(5):
        if hasattr(node, 'names'):
            return list(node.names)
        node = getattr(node, 'feature', None)
        if node is None:
            break
    raise RuntimeError('could not resolve COCO category names')

def build_index(force: bool=False) -> dict:
    if INDEX_PATH.exists() and (not force):
        return json.loads(INDEX_PATH.read_text())
    import pandas as pd
    names = _category_names()
    files = sorted(glob.glob(str(_COCO_REPO_DIR / 'snapshots' / '*' / 'data' / '*.parquet')))
    if not files:
        raise FileNotFoundError('detection-datasets/coco parquet not found; download first.')
    index: dict[str, list] = {}
    for f in files:
        df = pd.read_parquet(f, columns=['image_id', 'objects'])
        for image_id, objs in zip(df['image_id'].tolist(), df['objects'].tolist()):
            cats = objs['category']
            boxes = objs['bbox']
            recs = []
            for c, b in zip(cats, boxes):
                nm = names[int(c)] if 0 <= int(c) < len(names) else str(c)
                recs.append({'name': nm, 'bbox_xyxy': [float(v) for v in b]})
            index[str(int(image_id))] = recs
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index))
    return index

class COCOBoxes:

    def __init__(self):
        self._idx: Optional[dict] = None

    def available(self) -> bool:
        return INDEX_PATH.exists()

    def _ensure(self):
        if self._idx is None:
            self._idx = json.loads(INDEX_PATH.read_text()) if INDEX_PATH.exists() else {}

    def boxes_for(self, coco_image_id: str) -> list[dict]:
        self._ensure()
        return self._idx.get(str(int(coco_image_id)), [])

def coco_id_from_source(image_source: str) -> Optional[str]:
    if not image_source:
        return None
    tail = image_source.rsplit('_', 1)[-1]
    try:
        return str(int(tail))
    except ValueError:
        return None
