from __future__ import annotations
import copy
import importlib
import os
from typing import Any
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from .base import VLMAdapter, SYSTEM_PROMPT
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

def _build_transform(input_size: int):
    return T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float('inf')
    best = (1, 1)
    area = width * height
    for r in target_ratios:
        tar = r[0] / r[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]):
            best_diff = diff
            best = r
    return best

def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    ratio = _find_closest_aspect_ratio(ar, target_ratios, w, h, image_size)
    tw, th = (image_size * ratio[0], image_size * ratio[1])
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = []
    cols = tw // image_size
    for i in range(blocks):
        box = (i % cols * image_size, i // cols * image_size, (i % cols + 1) * image_size, (i // cols + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

class InternVLAdapter(VLMAdapter):
    family = 'internvl'

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.repo, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(self.spec.repo, dtype=self.dtype, device_map={'': self.device}, trust_remote_code=True, attn_implementation='eager')
        cfg = self.model.config
        self.image_size = int(getattr(cfg, 'force_image_size', None) or cfg.vision_config.image_size)
        self.use_thumbnail = bool(getattr(cfg, 'use_thumbnail', True))
        self.max_tiles = int(getattr(cfg, 'max_dynamic_patch', 12) or 12)
        _cap = os.environ.get('VISTA_MAX_TILES')
        if _cap:
            self.max_tiles = min(self.max_tiles, int(_cap))
        self.num_image_token = int(self.model.num_image_token)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._get_conv_template = self._resolve_conv_fn()
        self._mark_loaded()

    def _resolve_conv_fn(self):
        try:
            mod = importlib.import_module(type(self.model).__module__)
            return getattr(mod, 'get_conv_template')
        except Exception:
            return None

    def _pixel_values(self, image: Image.Image) -> torch.Tensor:
        tiles = _dynamic_preprocess(image, max_num=self.max_tiles, image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        tf = _build_transform(self.image_size)
        pv = torch.stack([tf(t) for t in tiles])
        return pv.to(device=self.model.device, dtype=self.dtype)

    def _fresh_template(self):
        if self._get_conv_template is not None:
            tmpl = self._get_conv_template(self.model.template)
        else:
            tmpl = copy.deepcopy(self.model.conv_template)
            tmpl.messages = []
        tmpl.system_message = self.model.system_message or SYSTEM_PROMPT
        return tmpl

    def _encode(self, image: Image.Image, turns: list[tuple[int, str]], add_generation_prompt: bool) -> dict[str, Any]:
        pixel_values = self._pixel_values(image)
        num_patches = pixel_values.shape[0]
        tmpl = self._fresh_template()
        for role_idx, text in turns:
            tmpl.append_message(tmpl.roles[role_idx], text)
        if add_generation_prompt:
            tmpl.append_message(tmpl.roles[1], None)
        query = tmpl.get_prompt()
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
        enc = self.tokenizer(query, return_tensors='pt')
        return {'input_ids': enc['input_ids'].to(self.model.device), 'attention_mask': enc['attention_mask'].to(self.model.device), 'pixel_values': pixel_values}

    def _forward_extra(self, inputs) -> dict[str, Any]:
        pv = inputs.get('pixel_values')
        n = pv.shape[0] if pv is not None else 1
        return {'image_flags': torch.ones((n, 1), dtype=torch.long, device=pv.device)}

    def build_inputs(self, image: Image.Image, prompt_text: str) -> dict[str, Any]:
        return self._encode(image, [(0, '<image>\n' + prompt_text)], add_generation_prompt=True)

    def build_inputs_batch(self, images, prompt_text):
        pvs = [self._pixel_values(im) for im in images]
        tiles = [pv.shape[0] for pv in pvs]
        if len(set(tiles)) != 1:
            raise NotImplementedError
        num_patches = tiles[0]
        pixel_values = torch.cat(pvs, dim=0)
        tmpl = self._fresh_template()
        tmpl.append_message(tmpl.roles[0], '<image>\n' + prompt_text)
        tmpl.append_message(tmpl.roles[1], None)
        query = tmpl.get_prompt()
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
        enc = self.tokenizer([query] * len(images), return_tensors='pt')
        return {'input_ids': enc['input_ids'].to(self.model.device), 'attention_mask': enc['attention_mask'].to(self.model.device), 'pixel_values': pixel_values}

    def _forward_extra_batch(self, inputs, n):
        pv = inputs.get('pixel_values')
        total = pv.shape[0] if pv is not None else n
        return {'image_flags': torch.ones((total, 1), dtype=torch.long, device=pv.device)}

    def _qa_text(self, question, format_instruction):
        instr = format_instruction or 'Answer the question using a single word or short phrase.'
        return f'<image>\n{question}\n{instr}'

    def build_verbalized_inputs(self, image, question, answer, format_instruction=None):
        from .base import VERBALIZED_PROMPT
        return self._encode(image, [(0, self._qa_text(question, format_instruction)), (1, answer), (0, VERBALIZED_PROMPT)], add_generation_prompt=True)

    def build_forward_inputs(self, image, question, answer, format_instruction=None):
        return self._encode(image, [(0, self._qa_text(question, format_instruction)), (1, answer)], add_generation_prompt=False)
