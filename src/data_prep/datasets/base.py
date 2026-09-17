from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from PIL import Image

@dataclass
class NativeRegion:
    name: str
    bbox_xyxy: tuple[float, float, float, float]
    source: str
    mask: Optional[np.ndarray] = None

@dataclass
class DatasetItem:
    dataset: str
    split: str
    item_id: str
    image_key: str
    question: str
    answer: str
    answer_type: str
    lang: str = 'en'
    relevant_hint: Optional[str] = None
    relevant_bbox: Optional[tuple] = None
    meta: dict = field(default_factory=dict)

    def manifest_row(self) -> dict:
        return {'dataset': self.dataset, 'split': self.split, 'item_id': self.item_id, 'image_key': self.image_key, 'answer_type': self.answer_type, 'lang': self.lang, 'has_relevant_hint': self.relevant_hint is not None, 'has_native_bbox': self.relevant_bbox is not None}

    def to_dict(self) -> dict:
        return {'dataset': self.dataset, 'split': self.split, 'item_id': self.item_id, 'image_key': self.image_key, 'question': self.question, 'answer': self.answer, 'answer_type': self.answer_type, 'lang': self.lang, 'relevant_hint': self.relevant_hint, 'relevant_bbox': list(self.relevant_bbox) if self.relevant_bbox else None, 'meta': self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> 'DatasetItem':
        bbox = d.get('relevant_bbox')
        return cls(dataset=d['dataset'], split=d['split'], item_id=d['item_id'], image_key=d['image_key'], question=d['question'], answer=d['answer'], answer_type=d['answer_type'], lang=d.get('lang', 'en'), relevant_hint=d.get('relevant_hint'), relevant_bbox=tuple(bbox) if bbox else None, meta=d.get('meta', {}))

class DatasetAdapter(abc.ABC):
    name: str = 'base'

    @abc.abstractmethod
    def splits(self) -> list[str]:
        ...

    @abc.abstractmethod
    def load_split(self, split: str, limit: Optional[int]=None) -> list[DatasetItem]:
        ...

    @abc.abstractmethod
    def get_image(self, item: DatasetItem) -> Image.Image:
        ...

    def native_regions(self, item: DatasetItem) -> list[NativeRegion]:
        return []

    def stratify_key(self, item: DatasetItem) -> str:
        return item.answer_type
