from __future__ import annotations
import torch
from src.config import get_model_spec
from .internvl import InternVLAdapter
from .gemma3 import Gemma3Adapter
from .molmo import Molmo2Adapter
from .llava_onevision import LlavaOneVisionAdapter
_FAMILY = {'internvl': InternVLAdapter, 'gemma3': Gemma3Adapter, 'molmo': Molmo2Adapter, 'llava_onevision': LlavaOneVisionAdapter}

def build_adapter(key: str, device: str='cuda:0', dtype: torch.dtype=torch.bfloat16):
    spec = get_model_spec(key)
    cls = _FAMILY[spec.family]
    return cls(spec, device=device, dtype=dtype)
