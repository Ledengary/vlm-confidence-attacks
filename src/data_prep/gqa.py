from __future__ import annotations
import ast
import functools
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
from src.config import GQA_DIR
HF_HUB = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'
GQA_DATASET_DIR = HF_HUB / 'datasets--lmms-lab--GQA'

@dataclass
class GQAObject:
    obj_id: str
    name: str
    x: int
    y: int
    w: int
    h: int
    attributes: list[str] = field(default_factory=list)

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def area(self) -> int:
        return int(self.w) * int(self.h)

@dataclass
class GQAItem:
    qid: str
    image_id: str
    question: str
    answer: str
    full_answer: str
    relevant_obj_id: Optional[str]
    semantic_str: str
    types: dict
    raw: dict

def _instructions_parquet(split: str, balanced: bool) -> Path:
    tag = f"{split}_{('balanced' if balanced else 'all')}_instructions"
    hits = glob.glob(str(GQA_DATASET_DIR / 'snapshots' / '*' / tag / '*.parquet'))
    if not hits:
        raise FileNotFoundError(f'GQA instructions parquet not found for {tag}. Ensure the dataset is cached.')
    return Path(hits[0])

def _coerce_annotations(ann: Any) -> dict:
    out: dict[str, list[dict]] = {}
    if ann is None:
        return out
    if isinstance(ann, dict):
        for k, v in ann.items():
            items = []
            seq = v.tolist() if isinstance(v, np.ndarray) else v
            for entry in seq or []:
                if isinstance(entry, dict):
                    items.append({'objectId': str(entry.get('objectId')), 'value': str(entry.get('value'))})
            out[k] = items
    return out

def relevant_object_id(row: dict) -> Optional[str]:
    ann = _coerce_annotations(row.get('annotations'))
    for key in ('answer', 'question', 'fullAnswer'):
        for entry in ann.get(key, []):
            val = entry.get('value')
            if val and val != 'None':
                return val
    sem = row.get('semanticStr') or ''
    import re
    m = re.search('\\((\\d+)\\)', sem)
    return m.group(1) if m else None
_NON_OBJECT_SELECTS = {'scene', 'image', ''}

def relevant_object_name(row: dict) -> Optional[str]:
    sem = row.get('semanticStr') or ''
    import re
    m = re.search('select:\\s*([^(>\\[]+)', sem)
    if m:
        name = m.group(1).strip().strip(' .')
        name = re.sub('\\s*\\(\\d+\\)\\s*$', '', name)
        name = re.sub('[\\s\\-]+$', '', name).strip()
        if name and name.lower() not in _NON_OBJECT_SELECTS:
            return name
    return None

def load_items(split: str='val', balanced: bool=True, limit: Optional[int]=None) -> list[GQAItem]:
    df = pd.read_parquet(_instructions_parquet(split, balanced))
    if limit is not None:
        df = df.head(limit)
    items: list[GQAItem] = []
    for _, r in df.iterrows():
        row = r.to_dict()
        items.append(GQAItem(qid=str(row.get('id')), image_id=str(row.get('imageId')), question=str(row.get('question')), answer=str(row.get('answer')), full_answer=str(row.get('fullAnswer')), relevant_obj_id=relevant_object_id(row), semantic_str=str(row.get('semanticStr')), types=row.get('types') if isinstance(row.get('types'), dict) else {}, raw=row))
    return items

@functools.lru_cache(maxsize=4)
def load_scene_graphs(split: str='val') -> dict:
    path = GQA_DIR / f'{split}_sceneGraphs.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Download official GQA sceneGraphs.zip into {GQA_DIR}.')
    with open(path) as f:
        return json.load(f)

def objects_for_image(image_id: str, split: str='val') -> dict[str, GQAObject]:
    sg = load_scene_graphs(split).get(str(image_id))
    if not sg:
        return {}
    out: dict[str, GQAObject] = {}
    for oid, o in (sg.get('objects') or {}).items():
        out[str(oid)] = GQAObject(obj_id=str(oid), name=str(o.get('name', '')), x=int(o.get('x', 0)), y=int(o.get('y', 0)), w=int(o.get('w', 0)), h=int(o.get('h', 0)), attributes=list(o.get('attributes', []) or []))
    return out

def object_bbox_name(image_id: str, obj_id: str, split: str='val'):
    sg = load_scene_graphs(split).get(str(image_id))
    if not sg:
        return (None, None)
    o = (sg.get('objects') or {}).get(str(obj_id))
    if not o:
        return (None, None)
    x, y, w, h = (int(o.get('x', 0)), int(o.get('y', 0)), int(o.get('w', 0)), int(o.get('h', 0)))
    return ((x, y, x + w, y + h), str(o.get('name', '')))

def image_size(image_id: str, split: str='val') -> Optional[tuple[int, int]]:
    sg = load_scene_graphs(split).get(str(image_id))
    if sg and 'width' in sg and ('height' in sg):
        return (int(sg['width']), int(sg['height']))
    return None
GQA_IMG_DIR = GQA_DIR / 'images'

def extract_images_to_disk(split: str='train', balanced: bool=True) -> int:
    import glob as _glob
    tag = f"{split}_{('balanced' if balanced else 'all')}_images"
    hits = sorted(_glob.glob(str(GQA_DATASET_DIR / 'snapshots' / '*' / tag / '*.parquet')))
    out = GQA_IMG_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for h in hits:
        df = pd.read_parquet(h)
        id_col = 'id' if 'id' in df.columns else df.columns[0]
        for iid, cell in zip(df[id_col].tolist(), df['image'].tolist()):
            p = out / f'{iid}.jpg'
            if p.exists():
                n += 1
                continue
            data = cell.get('bytes') if isinstance(cell, dict) else cell
            if isinstance(data, (bytes, bytearray)):
                p.write_bytes(data)
                n += 1
    return n

class GQAImages:

    def __init__(self, split: str='val', balanced: bool=True):
        self.split = split
        self.balanced = balanced
        self._df = None
        self._index: dict[str, int] = {}
        self._disk = GQA_IMG_DIR / split

    def _ensure(self):
        if self._df is not None:
            return
        tag = f"{self.split}_{('balanced' if self.balanced else 'all')}_images"
        hits = sorted(glob.glob(str(GQA_DATASET_DIR / 'snapshots' / '*' / tag / '*.parquet')))
        if not hits:
            raise FileNotFoundError(f"GQA images parquet not found for {tag}. Download it with: hf download lmms-lab/GQA --repo-type dataset --include '{tag}/*'")
        self._df = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
        id_col = 'id' if 'id' in self._df.columns else self._df.columns[0]
        self._id_col = id_col
        self._index = {str(v): i for i, v in enumerate(self._df[id_col].tolist())}

    def __contains__(self, image_id: str) -> bool:
        if (self._disk / f'{image_id}.jpg').exists():
            return True
        self._ensure()
        return str(image_id) in self._index

    def get(self, image_id: str) -> Image.Image:
        p = self._disk / f'{image_id}.jpg'
        if p.exists():
            return Image.open(p).convert('RGB')
        self._ensure()
        idx = self._index[str(image_id)]
        cell = self._df.iloc[idx]['image']
        return _decode_image(cell)

def _decode_image(cell: Any) -> Image.Image:
    import io
    if isinstance(cell, dict):
        if cell.get('bytes'):
            return Image.open(io.BytesIO(cell['bytes'])).convert('RGB')
        if cell.get('path'):
            return Image.open(cell['path']).convert('RGB')
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert('RGB')
    if isinstance(cell, Image.Image):
        return cell.convert('RGB')
    raise ValueError(f'Unrecognized image cell type: {type(cell)}')
