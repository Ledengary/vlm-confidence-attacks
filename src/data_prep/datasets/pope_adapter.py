from __future__ import annotations
import glob
import io
import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image
from src.config import DATA_DIR
from .base import DatasetAdapter, DatasetItem, NativeRegion
from .coco_boxes import COCOBoxes, coco_id_from_source
_HUB = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'
_POPE_DIR = _HUB / 'datasets--lmms-lab--POPE'
_IMG_DISK = DATA_DIR / 'pope' / 'images'
_OBJ_PAT = re.compile('is there\\s+(?:a|an|the)?\\s*(.+?)\\s+in the image', re.IGNORECASE)

class POPEAdapter(DatasetAdapter):
    name = 'pope'

    def __init__(self):
        self._df = None
        self._coco = COCOBoxes()

    def splits(self) -> list[str]:
        return ['test']

    def _load_df(self):
        if self._df is None:
            files = sorted(glob.glob(str(_POPE_DIR / 'snapshots' / '*' / 'data' / '*.parquet')))
            self._df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return self._df

    @staticmethod
    def _parse_object(question: str) -> Optional[str]:
        m = _OBJ_PAT.search(question or '')
        return m.group(1).strip() if m else None

    def load_split(self, split: str, limit: Optional[int]=None) -> list[DatasetItem]:
        df = self._load_df()
        if limit is not None:
            df = df.head(limit)
        items = []
        for i, r in df.iterrows():
            src = str(r.get('image_source'))
            items.append(DatasetItem(dataset=self.name, split=split, item_id=f"pope_{r.get('category')}_{r.get('question_id')}_{i}", image_key=str(i), question=str(r.get('question')), answer=str(r.get('answer')), answer_type=str(r.get('category')), lang='en', relevant_hint=self._parse_object(str(r.get('question'))), relevant_bbox=None, meta={'image_source': src, 'coco_image_id': coco_id_from_source(src), 'question_id': int(r.get('question_id'))}))
        return items

    def extract_images_to_disk(self) -> int:
        _IMG_DISK.mkdir(parents=True, exist_ok=True)
        df = self._load_df()
        n = 0
        for i in range(len(df)):
            r = df.iloc[i]
            cid = coco_id_from_source(str(r.get('image_source')))
            if not cid:
                continue
            p = _IMG_DISK / f'{cid}.jpg'
            if p.exists():
                continue
            cell = r['image']
            data = cell.get('bytes') if isinstance(cell, dict) else cell
            if isinstance(data, (bytes, bytearray)):
                p.write_bytes(data)
                n += 1
        return n

    def get_image(self, item: DatasetItem) -> Image.Image:
        cid = item.meta.get('coco_image_id')
        if cid:
            p = _IMG_DISK / f'{cid}.jpg'
            if p.exists():
                return Image.open(p).convert('RGB')
        df = self._load_df()
        cell = df.iloc[int(item.image_key)]['image']
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
