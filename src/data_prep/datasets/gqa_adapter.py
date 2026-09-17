from __future__ import annotations
from typing import Optional
from PIL import Image
from src.data_prep import gqa
from .base import DatasetAdapter, DatasetItem, NativeRegion
_SPLIT_TO_TAG = {'train': 'train', 'val': 'val', 'testdev': 'testdev'}

class GQAAdapter(DatasetAdapter):
    name = 'gqa'

    def __init__(self):
        self._images: dict[str, gqa.GQAImages] = {}

    def splits(self) -> list[str]:
        return ['train', 'val', 'testdev']

    def _has_scene_graph(self, split: str) -> bool:
        return split in ('train', 'val')

    def load_split(self, split: str, limit: Optional[int]=None) -> list[DatasetItem]:
        rows = gqa.load_items(split=split, balanced=True, limit=limit)
        items: list[DatasetItem] = []
        for it in rows:
            row = it.raw
            structural = (it.types or {}).get('structural', 'unknown')
            name = gqa.relevant_object_name(row)
            bbox = None
            if self._has_scene_graph(split) and it.relevant_obj_id:
                bbox, sg_name = gqa.object_bbox_name(it.image_id, it.relevant_obj_id, split=split)
                name = name or (sg_name or None)
            items.append(DatasetItem(dataset=self.name, split=split, item_id=f'gqa_{split}_{it.qid}', image_key=it.image_id, question=it.question, answer=it.answer, answer_type=structural, lang='en', relevant_hint=name, relevant_bbox=bbox, meta={'qid': it.qid, 'semantic_type': (it.types or {}).get('semantic'), 'relevant_obj_id': it.relevant_obj_id}))
        return items

    def get_image(self, item: DatasetItem) -> Image.Image:
        store = self._images.get(item.split)
        if store is None:
            store = gqa.GQAImages(split=item.split, balanced=True)
            self._images[item.split] = store
        return store.get(item.image_key)

    def native_regions(self, item: DatasetItem) -> list[NativeRegion]:
        if not self._has_scene_graph(item.split):
            return []
        out = []
        for oid, o in gqa.objects_for_image(item.image_key, split=item.split).items():
            out.append(NativeRegion(name=o.name, bbox_xyxy=o.bbox_xyxy, source='gqa_scene_graph'))
        return out
