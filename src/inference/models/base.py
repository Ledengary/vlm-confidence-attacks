from __future__ import annotations
import abc
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
import torch
from PIL import Image
SYSTEM_PROMPT = 'You are a vision language assistant. Provide brief, complete answers.'
VERBALIZED_PROMPT = 'Provide the probability that your answer is correct. Give ONLY the probability, no other words or explanation. For example: Probability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>'
PRIMARY_REDUCTION = 'geomean'

def reduce_token_probs(probs: list[float]) -> dict[str, float]:
    if not probs:
        nan = float('nan')
        return {'geomean': nan, 'mean': nan, 'product': nan, 'min': nan, 'first': nan, 'n_tokens': 0}
    t = torch.tensor(probs, dtype=torch.float64).clamp_min(1e-12)
    logp = t.log()
    return {'geomean': float(logp.mean().exp()), 'mean': float(t.mean()), 'product': float(logp.sum().exp()), 'min': float(t.min()), 'first': float(t[0]), 'n_tokens': len(probs)}

@dataclass
class AnswerResult:
    answer_text: str
    token_ids: list[int]
    token_strs: list[str]
    token_probs: list[float]
    confidence: dict[str, float]
    raw_text: str

    @property
    def token_prob_confidence(self) -> float:
        return self.confidence.get(PRIMARY_REDUCTION, float('nan'))

@dataclass
class VerbalizedResult:
    value: Optional[float]
    raw_text: str
    parsed_ok: bool = False

class VLMAdapter(abc.ABC):
    family: str = 'base'

    def __init__(self, spec, device: str='cuda:0', dtype: torch.dtype=torch.bfloat16):
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False

    @abc.abstractmethod
    def load(self) -> None:
        ...

    def _mark_loaded(self) -> None:
        self.model.eval()
        if self.tokenizer is None:
            self.tokenizer = getattr(self.processor, 'tokenizer', None) or self.tokenizer
        self._loaded = True

    def commit_sha(self) -> str:
        try:
            cfg = self.model.config
            sha = getattr(cfg, '_commit_hash', None)
            if sha:
                return sha
        except Exception:
            pass
        return 'unknown'

    @abc.abstractmethod
    def build_inputs(self, image: Image.Image, prompt_text: str) -> dict[str, Any]:
        ...

    @torch.inference_mode()
    def _generate(self, inputs: dict[str, Any], max_new_tokens: int=64) -> dict[str, Any]:
        gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, output_scores=True, return_dict_in_generate=True, pad_token_id=self._pad_id())
        input_len = inputs['input_ids'].shape[1]
        seq = gen.sequences[0]
        scores = gen.scores
        n_new = len(scores) if scores is not None else max(0, seq.shape[0] - input_len)
        new_ids = seq[-n_new:] if n_new > 0 else seq[input_len:]
        return {'new_ids': new_ids, 'scores': scores}

    def _pad_id(self):
        tok = self.tokenizer
        if tok is None:
            return None
        return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    _END_MARKERS = ('<end_of_turn>', '<|im_end|>', '<|endoftext|>', '</s>', '<|eot_id|>', '<eos>', '<|end|>')

    def _special_ids(self) -> set[int]:
        tok = self.tokenizer
        if tok is None:
            return set()
        ids: set[int] = set(tok.all_special_ids or [])
        for attr in ('eos_token_id', 'pad_token_id', 'bos_token_id'):
            v = getattr(tok, attr, None)
            if isinstance(v, int):
                ids.add(v)
        ids.update(getattr(tok, 'additional_special_tokens_ids', []) or [])
        for marker in self._END_MARKERS:
            try:
                mid = tok.convert_tokens_to_ids(marker)
                if isinstance(mid, int) and mid is not None and (mid >= 0) and (mid != getattr(tok, 'unk_token_id', -1)):
                    ids.add(mid)
            except Exception:
                pass
        return ids

    def _span_probs(self, new_ids: torch.Tensor, scores):
        specials = self._special_ids()
        ids: list[int] = []
        strs: list[str] = []
        probs: list[float] = []
        for step, tok_id in enumerate(new_ids.tolist()):
            if step >= len(scores):
                break
            if tok_id in specials:
                continue
            logits = scores[step][0].to(torch.float32)
            p = torch.softmax(logits, dim=-1)[tok_id].item()
            ids.append(int(tok_id))
            strs.append(self.tokenizer.decode([tok_id]))
            probs.append(float(p))
        return (ids, strs, probs)

    def generate_answer(self, image: Image.Image, question: str, format_instruction: Optional[str]=None, max_new_tokens: int=64) -> AnswerResult:
        inputs = self.build_inputs(image, self.answer_prompt(question, format_instruction))
        out = self._generate(inputs, max_new_tokens=max_new_tokens)
        ids, strs, probs = self._span_probs(out['new_ids'], out['scores'])
        raw = self.tokenizer.decode(out['new_ids'], skip_special_tokens=True).strip()
        return AnswerResult(answer_text=self.parse_answer(raw), token_ids=ids, token_strs=strs, token_probs=probs, confidence=reduce_token_probs(probs), raw_text=raw)

    def verbalized_confidence(self, image: Image.Image, question: str, answer: str, format_instruction: Optional[str]=None, max_new_tokens: int=24) -> VerbalizedResult:
        inputs = self.build_verbalized_inputs(image, question, answer, format_instruction)
        out = self._generate(inputs, max_new_tokens=max_new_tokens)
        raw = self.tokenizer.decode(out['new_ids'], skip_special_tokens=True).strip()
        value = parse_verbalized_confidence(raw)
        return VerbalizedResult(value=value, raw_text=raw, parsed_ok=value is not None)

    @torch.inference_mode()
    def score_answer_ids(self, image: Image.Image, question: str, answer_token_ids: list[int], format_instruction: Optional[str]=None) -> dict[str, float]:
        if not answer_token_ids:
            return reduce_token_probs([])
        inputs = self.build_inputs(image, self.answer_prompt(question, format_instruction))
        prompt_ids = inputs['input_ids']
        L = prompt_ids.shape[1]
        ans = torch.tensor([list(answer_token_ids)], device=prompt_ids.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, ans], dim=1)
        inputs2 = dict(inputs)
        inputs2['input_ids'] = full
        if inputs2.get('attention_mask') is not None:
            inputs2['attention_mask'] = torch.ones_like(full)
        _extend_seq_aux(inputs2, L, len(answer_token_ids))
        inputs2.update(self._forward_extra(inputs2))
        out = self.model(**inputs2, use_cache=False)
        logits = out.logits[0].to(torch.float32)
        probs = []
        for i, tid in enumerate(answer_token_ids):
            probs.append(float(torch.softmax(logits[L - 1 + i], dim=-1)[tid].item()))
        return reduce_token_probs(probs)

    @torch.inference_mode()
    def score_answer_ids_batch(self, images: list, question: str, answer_token_ids: list[int], format_instruction: Optional[str]=None, sub_batch: int=6) -> list[dict]:
        if not answer_token_ids or not images:
            return [reduce_token_probs([]) for _ in images]
        out: list[dict] = []
        for i in range(0, len(images), sub_batch):
            out.extend(self._score_chunk(images[i:i + sub_batch], question, answer_token_ids, format_instruction))
        return out

    def _score_chunk(self, images, question, answer_token_ids, format_instruction):
        try:
            inputs = self.build_inputs_batch(images, self.answer_prompt(question, format_instruction))
        except NotImplementedError:
            return [self.score_answer_ids(im, question, answer_token_ids, format_instruction) for im in images]
        prompt_ids = inputs['input_ids']
        n, L = prompt_ids.shape
        ans = torch.tensor([list(answer_token_ids)], device=prompt_ids.device, dtype=prompt_ids.dtype).repeat(n, 1)
        full = torch.cat([prompt_ids, ans], dim=1)
        inputs2 = dict(inputs)
        inputs2['input_ids'] = full
        if inputs2.get('attention_mask') is not None:
            inputs2['attention_mask'] = torch.ones_like(full)
        _extend_seq_aux(inputs2, L, len(answer_token_ids))
        inputs2.update(self._forward_extra_batch(inputs2, n))
        logits = self.model(**inputs2, use_cache=False).logits.to(torch.float32)
        res = []
        for r in range(n):
            probs = [float(torch.softmax(logits[r, L - 1 + j], dim=-1)[tid].item()) for j, tid in enumerate(answer_token_ids)]
            res.append(reduce_token_probs(probs))
        return res

    def build_inputs_batch(self, images, prompt_text):
        raise NotImplementedError

    def _forward_extra_batch(self, inputs, n) -> dict:
        return {}

    @torch.inference_mode()
    def answer_hidden_state(self, image: Image.Image, question: str, answer: str, format_instruction: Optional[str]=None):
        inputs = self.build_forward_inputs(image, question, answer, format_instruction)
        inputs.update(self._forward_extra(inputs))
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1]
        return hs[0, -1].to(torch.float32).cpu().numpy()

    def answer_hidden_all_layers(self, image: Image.Image, question: str, answer: str, format_instruction: Optional[str]=None):
        inputs = self.build_forward_inputs(image, question, answer, format_instruction)
        inputs.update(self._forward_extra(inputs))
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        return torch.stack([h[0, -1] for h in out.hidden_states]).detach().to(torch.float16).cpu().numpy()

    @torch.inference_mode()
    def answer_hidden_pik_saplma(self, image: Image.Image, question: str, answer_token_ids: list[int], format_instruction: Optional[str]=None):
        if not answer_token_ids:
            return (None, None)
        inputs = self.build_inputs(image, self.answer_prompt(question, format_instruction))
        prompt_ids = inputs['input_ids']
        L = int(prompt_ids.shape[1])
        ans = torch.tensor([list(answer_token_ids)], device=prompt_ids.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, ans], dim=1)
        inputs2 = dict(inputs)
        inputs2['input_ids'] = full
        if inputs2.get('attention_mask') is not None:
            inputs2['attention_mask'] = torch.ones_like(full)
        _extend_seq_aux(inputs2, L, len(answer_token_ids))
        inputs2.update(self._forward_extra(inputs2))
        out = self.model(**inputs2, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1][0]
        pik = hs[L - 1].to(torch.float16).cpu().numpy()
        saplma = hs[-1].to(torch.float16).cpu().numpy()
        return (pik, saplma)

    def _forward_extra(self, inputs) -> dict:
        return {}

    def build_verbalized_inputs(self, image, question, answer, format_instruction=None):
        raise NotImplementedError

    def build_forward_inputs(self, image, question, answer, format_instruction=None):
        raise NotImplementedError

    def answer_prompt(self, question: str, format_instruction: Optional[str]=None) -> str:
        instr = format_instruction or 'Answer the question using a single word or short phrase.'
        return f'{question}\n{instr}'

    def parse_answer(self, raw: str) -> str:
        m = re.search('answer\\s*[:\\-]\\s*(.+)', raw, flags=re.IGNORECASE)
        if m:
            return m.group(1).splitlines()[0].strip(' ."\'')
        return raw.splitlines()[0].strip(' ."\'') if raw else ''

class ProcessorAdapter(VLMAdapter):
    model_class_name: str = 'auto'

    def load(self) -> None:
        import transformers
        from transformers import AutoProcessor
        trc = self.spec.trust_remote_code
        self.processor = AutoProcessor.from_pretrained(self.spec.repo, trust_remote_code=trc)
        if self.model_class_name == 'auto':
            from transformers import AutoModelForImageTextToText as ModelCls
        else:
            ModelCls = getattr(transformers, self.model_class_name)
        attn_impl = 'eager' if os.environ.get('VISTA_EAGER_ATTN') == '1' else 'sdpa'
        self.model = ModelCls.from_pretrained(self.spec.repo, dtype=self.dtype, device_map={'': self.device}, trust_remote_code=trc, attn_implementation=attn_impl)
        self._mark_loaded()

    def _messages(self, prompt_text: str) -> list[dict]:
        return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]}, {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt_text}]}]

    def build_inputs(self, image: Image.Image, prompt_text: str) -> dict[str, Any]:
        messages = self._messages(prompt_text)
        prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt_str], images=[image], return_tensors='pt')
        return _to_device(inputs, self.model.device, self.dtype)

    def _qa_turn(self, question, format_instruction):
        instr = format_instruction or 'Answer the question using a single word or short phrase.'
        return {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': f'{question}\n{instr}'}]}

    def build_verbalized_inputs(self, image, question, answer, format_instruction=None):
        messages = [self._qa_turn(question, format_instruction), {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}, {'role': 'user', 'content': [{'type': 'text', 'text': VERBALIZED_PROMPT}]}]
        prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt_str], images=[image], return_tensors='pt')
        return _to_device(inputs, self.model.device, self.dtype)

    def build_forward_inputs(self, image, question, answer, format_instruction=None):
        messages = [self._qa_turn(question, format_instruction), {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}]
        prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        inputs = self.processor(text=[prompt_str], images=[image], return_tensors='pt')
        return _to_device(inputs, self.model.device, self.dtype)

    def build_inputs_batch(self, images, prompt_text):
        messages = self._messages(prompt_text)
        prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt_str] * len(images), images=list(images), return_tensors='pt', padding=True)
        return _to_device(inputs, self.model.device, self.dtype)

def _extend_seq_aux(inputs: dict, prompt_len: int, n_answer: int) -> None:
    for k, v in list(inputs.items()):
        if k in ('input_ids', 'attention_mask', 'pixel_values'):
            continue
        if isinstance(v, torch.Tensor) and v.dim() == 2 and (v.shape[1] == prompt_len) and (not v.is_floating_point()):
            pad = torch.zeros((v.shape[0], n_answer), dtype=v.dtype, device=v.device)
            inputs[k] = torch.cat([v, pad], dim=1)

def _to_device(inputs, device, dtype):
    out = {}
    data = inputs if isinstance(inputs, dict) else dict(inputs)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out
_CONF_PATS = [re.compile('confidence[:\\s]+(\\d{1,3}(?:\\.\\d+)?)', re.IGNORECASE), re.compile('(\\d{1,3}(?:\\.\\d+)?)\\s*%'), re.compile('(\\d{1,3}(?:\\.\\d+)?)\\s*percent', re.IGNORECASE)]
_NUM_PAT = re.compile('\\b(\\d{1,3}(?:\\.\\d+)?)\\b')

def parse_verbalized_confidence(text: str) -> Optional[float]:
    if not text:
        return None
    matches: list[tuple[int, float]] = []
    for pat in _CONF_PATS:
        for m in pat.finditer(text):
            v = float(m.group(1))
            if 0.0 <= v <= 100.0:
                matches.append((m.start(), v))
    if matches:
        matches.sort(key=lambda x: x[0])
        return _normalize_conf(matches[-1][1])
    nums = [float(x) for x in _NUM_PAT.findall(text) if 0.0 <= float(x) <= 100.0]
    if nums:
        return _normalize_conf(nums[-1])
    return None

def _normalize_conf(v: float) -> Optional[float]:
    if 0.0 <= v <= 1.0:
        return v
    if 0.0 <= v <= 100.0:
        return v / 100.0
    return None
