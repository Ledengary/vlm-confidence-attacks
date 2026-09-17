from __future__ import annotations
from .base import ProcessorAdapter

class LlavaOneVisionAdapter(ProcessorAdapter):
    family = 'llava_onevision'
    model_class_name = 'LlavaOnevisionForConditionalGeneration'
