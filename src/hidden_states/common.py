from __future__ import annotations
import re
import torch
from src.inference.models.base import _extend_seq_aux
SCORE_MAX_SIDE = 336

def _ds(img):
    w, h = img.size
    if max(w, h) <= SCORE_MAX_SIDE:
        return img
    s = SCORE_MAX_SIDE / max(w, h)
    return img.resize((int(w * s), int(h * s)))

def _score_inputs(adapter, image, question, ans_ids, spec):
    inputs = adapter.build_inputs(image, adapter.answer_prompt(question, spec.format_instruction))
    prompt_ids = inputs['input_ids']
    L = int(prompt_ids.shape[1])
    ans = torch.tensor([list(ans_ids)], device=prompt_ids.device, dtype=prompt_ids.dtype)
    full = torch.cat([prompt_ids, ans], dim=1)
    inputs2 = dict(inputs)
    inputs2['input_ids'] = full
    if inputs2.get('attention_mask') is not None:
        inputs2['attention_mask'] = torch.ones_like(full)
    _extend_seq_aux(inputs2, L, len(ans_ids))
    inputs2.update(adapter._forward_extra(inputs2))
    return (inputs2, L)

def _inject_pv(adapter, inputs, delta):
    inp = dict(inputs)
    dt = inp['pixel_values'].dtype
    inp['pixel_values'] = (inp['pixel_values'].float() + delta.float().to(inp['pixel_values'].device)).to(dt)
    return inp
_ARTICLES = {'a', 'an', 'the'}

def normalize_answer(s: str) -> str:
    s = (s or '').lower().strip()
    s = re.sub('[^\\w\\s]', ' ', s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return ' '.join(toks)

def exact_match(pred: str, gold: str) -> bool:
    p, g = (normalize_answer(pred), normalize_answer(gold))
    if not g:
        return False
    if p == g:
        return True
    return bool(re.search(f'\\b{re.escape(g)}\\b', p))

def score_correct(question: str, gold: str, pred: str) -> bool:
    return exact_match(pred, gold)
