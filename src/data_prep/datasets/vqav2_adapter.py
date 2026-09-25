from __future__ import annotations
import glob
import io
import os
from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
from src.config import DATA_DIR
from .base import DatasetAdapter, DatasetItem, NativeRegion
from .coco_boxes import COCOBoxes
_HUB = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'
_VQA_DIR = _HUB / 'datasets--lmms-lab--VQAv2'
_META_COLS = ['question_id', 'image_id', 'question', 'multiple_choice_answer', 'answer_type']
_IMG_DISK = DATA_DIR / 'vqav2' / 'images'

def extract_images_to_disk() -> int:
    _IMG_DISK.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(str(_VQA_DIR / 'snapshots' / '*' / 'data' / 'validation-*.parquet')))
    n = 0
    for f in shards:
        df = pd.read_parquet(f, columns=['image_id', 'image'])
        for iid, cell in zip(df['image_id'].tolist(), df['image'].tolist()):
            p = _IMG_DISK / f'{int(iid)}.jpg'
            if p.exists():
                continue
            data = cell.get('bytes') if isinstance(cell, dict) else cell
            if isinstance(data, (bytes, bytearray)):
                p.write_bytes(data)
                n += 1
    return n

class VQAv2Adapter(DatasetAdapter):
    name = 'vqav2'

    def __init__(self):
        self._shards = None
        self._img_cache_idx = None
        self._img_cache = None
        self._coco = COCOBoxes()

    def splits(self) -> list[str]:
        return ['validation']

    def _shard_files(self):
        if self._shards is None:
            self._shards = sorted(glob.glob(str(_VQA_DIR / 'snapshots' / '*' / 'data' / 'validation-*.parquet')))
        return self._shards

    def load_split(self, split: str, limit: Optional[int]=None) -> list[DatasetItem]:
        items = []
        for si, f in enumerate(self._shard_files()):
            df = pd.read_parquet(f, columns=_META_COLS)
            for ri, r in df.iterrows():
                items.append(DatasetItem(dataset=self.name, split=split, item_id=f"vqav2_{r['question_id']}", image_key=f'{si}:{ri}', question=str(r['question']), answer=str(r['multiple_choice_answer']), answer_type=str(r['answer_type']), lang='en', relevant_hint=None, relevant_bbox=None, meta={'question_id': int(r['question_id']), 'coco_image_id': str(int(r['image_id']))}))
                if limit is not None and len(items) >= limit:
                    return items
        return items

    def get_image(self, item: DatasetItem) -> Image.Image:
        cid = item.meta.get('coco_image_id')
        if cid:
            p = _IMG_DISK / f'{int(cid)}.jpg'
            if p.exists():
                return Image.open(p).convert('RGB')
        si, ri = (int(x) for x in item.image_key.split(':'))
        if self._img_cache_idx != si:
            df = pd.read_parquet(self._shard_files()[si], columns=['image'])
            self._img_cache = df['image'].tolist()
            self._img_cache_idx = si
        cell = self._img_cache[ri]
        if isinstance(cell, dict) and cell.get('bytes'):
            return Image.open(io.BytesIO(cell['bytes'])).convert('RGB')
        if isinstance(cell, (bytes, bytearray)):
            return Image.open(io.BytesIO(cell)).convert('RGB')
        return cell.convert('RGB')

    def native_regions(self, item: DatasetItem) -> list[NativeRegion]:
        cid = item.meta.get('coco_image_id')
        if not cid or not self._coco.available():
            return []
        return [NativeRegion(name=b['name'], bbox_xyxy=tuple(b['bbox_xyxy']), source='coco') for b in self._coco.boxes_for(cid)]
