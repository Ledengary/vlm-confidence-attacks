from __future__ import annotations
import torch

def official_pv_fn_factory(adapter, row):
    from PIL import Image

    def fn(adv_u8: torch.Tensor):
        arr = adv_u8.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(arr, mode='RGB')
        if adapter.spec.family == 'internvl':
            from src.inference.models.internvl import _build_transform, _dynamic_preprocess
            tiles = _dynamic_preprocess(img, max_num=adapter.max_tiles, image_size=adapter.image_size, use_thumbnail=adapter.use_thumbnail)
            tf = _build_transform(adapter.image_size)
            return torch.stack([tf(t) for t in tiles])
        ip = getattr(adapter.processor, 'image_processor', adapter.processor)
        return ip(images=[img], return_tensors='pt')['pixel_values'].float()
    return fn
