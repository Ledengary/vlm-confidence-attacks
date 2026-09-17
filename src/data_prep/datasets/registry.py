from __future__ import annotations
from .gqa_adapter import GQAAdapter
from .vqav2_adapter import VQAv2Adapter
from .pope_adapter import POPEAdapter
_ADAPTERS = {'gqa': GQAAdapter, 'vqav2': VQAv2Adapter, 'pope': POPEAdapter}

def get_adapter(name: str):
    if name not in _ADAPTERS:
        raise KeyError(f'Unknown dataset {name!r}. Known: {list(_ADAPTERS)}')
    return _ADAPTERS[name]()

def all_dataset_names() -> list[str]:
    return list(_ADAPTERS)
