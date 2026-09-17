from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
MANIFESTS_DIR = REPO_ROOT / 'manifests'
TABLES_DIR = REPO_ROOT / 'tables'
GQA_DIR = DATA_DIR / 'gqa'
SEED = 23

@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    family: str
    trust_remote_code: bool = False
    approx_params_b: float = 0.0
    notes: str = ''
MODEL_REGISTRY: dict[str, ModelSpec] = {'internvl3_5_2b': ModelSpec(key='internvl3_5_2b', repo='OpenGVLab/InternVL3_5-2B-Instruct', family='internvl', trust_remote_code=True, approx_params_b=2.0, notes='Non-HF Instruct variant: model_type internvl_chat, custom modeling code.'), 'molmo2_4b': ModelSpec(key='molmo2_4b', repo='allenai/Molmo2-4B', family='molmo', trust_remote_code=True, approx_params_b=4.0, notes='Loads via remote code (no native transformers class).'), 'gemma3_12b': ModelSpec(key='gemma3_12b', repo='google/gemma-3-12b-it', family='gemma3', trust_remote_code=False, approx_params_b=12.0), 'llava_onevision_7b': ModelSpec(key='llava_onevision_7b', repo='llava-hf/llava-onevision-qwen2-7b-ov-hf', family='llava_onevision', trust_remote_code=False, approx_params_b=8.0, notes='LlavaOnevisionForConditionalGeneration (Qwen2-7B LM + SigLIP).')}

def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f'Unknown model key {key!r}. Known: {list(MODEL_REGISTRY)}')
    return MODEL_REGISTRY[key]
