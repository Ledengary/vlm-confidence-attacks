from __future__ import annotations
import torch
from src.config import MODEL_REGISTRY, get_model_spec
from . import diffpreproc as dp
from .pil_exact import BICUBIC

def _image_processor(processor):
    ip = getattr(processor, 'image_processor', None)
    return ip if ip is not None else processor

def _internvl_params_from_config(cfg, max_tiles_env=None):
    image_size = int(getattr(cfg, 'force_image_size', None) or cfg.vision_config.image_size)
    use_thumbnail = bool(getattr(cfg, 'use_thumbnail', True))
    max_tiles = int(getattr(cfg, 'max_dynamic_patch', 12) or 12)
    if max_tiles_env:
        max_tiles = min(max_tiles, int(max_tiles_env))
    return {'image_size': image_size, 'max_tiles': max_tiles, 'use_thumbnail': use_thumbnail, 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'resample': BICUBIC}

def _gemma_params(ip):
    size = ip.size
    return {'height': int(size['height']), 'width': int(size['width']), 'resample': int(ip.resample), 'mean': tuple(ip.image_mean), 'std': tuple(ip.image_std)}

def _llava_params(ip):
    size = ip.size
    return {'size_h': int(size['height']), 'size_w': int(size['width']), 'grid_pinpoints': [tuple(x) for x in ip.image_grid_pinpoints], 'resample': int(ip.resample), 'mean': tuple(ip.image_mean), 'std': tuple(ip.image_std)}

def _molmo_params(ip):
    size = ip.size
    return {'base_size': int(size['height']), 'patch': int(ip.patch_size), 'max_crops': int(ip.max_crops), 'margins': tuple(ip.overlap_margins), 'mean': tuple(ip.image_mean), 'std': tuple(ip.image_std)}

def _make(family: str, params: dict):
    if family == 'internvl':

        def pp(raw01):
            return dp.internvl_pp(raw01, params['image_size'], params['max_tiles'], params['use_thumbnail'], params['mean'], params['std'])
    elif family == 'gemma3':

        def pp(raw01):
            return dp.gemma3_pp(raw01, params['height'], params['width'], params['resample'], params['mean'], params['std'])
    elif family == 'llava_onevision':

        def pp(raw01):
            return dp.llava_ov_pp(raw01, params['size_h'], params['size_w'], params['grid_pinpoints'], params['resample'], params['mean'], params['std'])
    elif family == 'molmo':

        def pp(raw01):
            return dp.molmo2_pp(raw01, params['base_size'], params['patch'], params['max_crops'], params['margins'], params['mean'], params['std'])
    else:
        raise NotImplementedError(family)
    return pp

def build_pp(adapter):
    spec = adapter.spec
    fam = spec.family
    if fam == 'internvl':
        params = {'image_size': adapter.image_size, 'max_tiles': adapter.max_tiles, 'use_thumbnail': adapter.use_thumbnail, 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'resample': BICUBIC}
    else:
        ip = _image_processor(adapter.processor)
        params = {'gemma3': _gemma_params, 'llava_onevision': _llava_params, 'molmo': _molmo_params}[fam](ip)
    return (_make(fam, params), {'family': fam, 'params': _jsonable(params)})

def build_pp_cpu(key: str):
    spec = get_model_spec(key)
    if spec.family == 'internvl':
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(spec.repo, trust_remote_code=True)
        params = _internvl_params_from_config(cfg)
    else:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(spec.repo, trust_remote_code=spec.trust_remote_code)
        ip = _image_processor(proc)
        params = {'gemma3': _gemma_params, 'llava_onevision': _llava_params, 'molmo': _molmo_params}[spec.family](ip)
    return (_make(spec.family, params), {'family': spec.family, 'params': _jsonable(params)})

def _jsonable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            out[k] = [list(x) if isinstance(x, (list, tuple)) else x for x in v]
        elif isinstance(v, torch.Tensor):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
MODELS = ['internvl3_5_2b', 'molmo2_4b', 'gemma3_12b', 'llava_onevision_7b']
SETS = ['gqa_val_eval', 'vqav2_ood', 'pope_ood']
MODEL_LABEL = {'internvl3_5_2b': 'InternVL3.5-2B', 'molmo2_4b': 'Molmo2-4B', 'gemma3_12b': 'Gemma-3-12B', 'llava_onevision_7b': 'LLaVA-OneVision-7B'}
SET_LABEL = {'gqa_val_eval': 'GQA', 'vqav2_ood': 'VQAv2', 'pope_ood': 'POPE'}
assert all((m in MODEL_REGISTRY for m in MODELS))
