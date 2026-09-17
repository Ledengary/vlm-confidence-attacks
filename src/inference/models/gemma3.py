from __future__ import annotations
from .base import ProcessorAdapter

class Gemma3Adapter(ProcessorAdapter):
    family = 'gemma3'
    model_class_name = 'Gemma3ForConditionalGeneration'
