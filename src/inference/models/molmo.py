from __future__ import annotations
from typing import Any
from PIL import Image
from .base import ProcessorAdapter, SYSTEM_PROMPT, _to_device

class Molmo2Adapter(ProcessorAdapter):
    family = 'molmo'
    model_class_name = 'auto'

    def _messages(self, prompt_text: str) -> list[dict]:
        return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]}, {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt_text}]}]

    def build_inputs(self, image: Image.Image, prompt_text: str) -> dict[str, Any]:
        try:
            return super().build_inputs(image, prompt_text)
        except Exception:
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': f'{SYSTEM_PROMPT}\n{prompt_text}'}]}]
            prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[prompt_str], images=[image], return_tensors='pt')
            return _to_device(inputs, self.model.device, self.dtype)
