from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
import numpy as np
import torch
from PIL import Image
from src.config import DATA_DIR, SEED
from src.inference.models.base import _extend_seq_aux
from src.inference.models.factory import build_adapter
from .families import build_pp

def load_adapter(model_key: str, gpu: int):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    adapter = build_adapter(model_key, device=f'cuda:{gpu}')
    adapter.load()
    for p in adapter.model.parameters():
        p.requires_grad_(False)
    adapter.model.eval()
    pp, pp_meta = build_pp(adapter)
    return (adapter, pp, pp_meta)

def forward_config(adapter, pp_meta: dict) -> dict:
    cfg = adapter.model.config
    out = {'model_key': adapter.spec.key, 'hf_repo': adapter.spec.repo, 'family': adapter.spec.family, 'commit_sha': adapter.commit_sha(), 'dtype': str(adapter.dtype).replace('torch.', ''), 'attn_implementation': getattr(cfg, '_attn_implementation', None), 'trust_remote_code': bool(adapter.spec.trust_remote_code), 'use_cache_in_scoring': False, 'batch_size': 1, 'generation': {'do_sample': False, 'num_beams': 1, 'greedy': True}, 'env_VISTA_EAGER_ATTN': None, 'env_VISTA_MAX_TILES': None, 'preproc': pp_meta, 'seed': SEED}
    if adapter.spec.family == 'internvl':
        out['internvl_max_tiles'] = int(adapter.max_tiles)
        out['internvl_image_size'] = int(adapter.image_size)
        out['internvl_use_thumbnail'] = bool(adapter.use_thumbnail)
    return out

@dataclass
class ItemCtx:
    row: dict
    raw01: torch.Tensor
    inputs: dict
    L: int
    gen_ids: list
    ans_ids: list
    constraint_ids: list
    pv_official: torch.Tensor
    model_dtype: torch.dtype
    prefix_ok: bool
    fresh_answer: str = ''
    gen_prompt_inputs: dict = field(default_factory=dict)

def raw_from_png(path, device) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a.copy()).permute(2, 0, 1).to(device)

def _strip_specials(adapter, ids: list) -> list:
    specials = adapter._special_ids()
    return [int(t) for t in ids if int(t) not in specials]

def build_item(adapter, pp, row: dict, device) -> ItemCtx:
    png = DATA_DIR / row['clean_png']
    img = Image.open(png).convert('RGB')
    raw01 = raw_from_png(png, device)
    prompt_text = adapter.answer_prompt(row['question'], row['format_instruction'])
    gen_inputs = adapter.build_inputs(img, prompt_text)
    out = adapter._generate(gen_inputs, max_new_tokens=int(row['max_new_tokens']))
    gen_ids = [int(t) for t in out['new_ids'].tolist()]
    ans_ids = _strip_specials(adapter, gen_ids)
    prefix_ok = ans_ids == gen_ids[:len(ans_ids)]
    raw_text = adapter.tokenizer.decode(out['new_ids'], skip_special_tokens=True).strip()
    fresh_answer = adapter.parse_answer(raw_text)
    prompt_ids = gen_inputs['input_ids']
    L = int(prompt_ids.shape[1])
    tail = torch.tensor([ans_ids], device=prompt_ids.device, dtype=prompt_ids.dtype)
    full = torch.cat([prompt_ids, tail], dim=1)
    inputs = dict(gen_inputs)
    inputs['input_ids'] = full
    if inputs.get('attention_mask') is not None:
        inputs['attention_mask'] = torch.ones_like(full)
    _extend_seq_aux(inputs, L, len(ans_ids))
    inputs.update(adapter._forward_extra(inputs))
    pv_official = inputs['pixel_values'].detach().clone()
    model_dtype = pv_official.dtype
    n = len(ans_ids)
    constraint_ids = list(ans_ids)
    if prefix_ok and len(gen_ids) > n:
        constraint_ids = constraint_ids + [gen_ids[n]]
    return ItemCtx(row=row, raw01=raw01, inputs=inputs, L=L, gen_ids=gen_ids, ans_ids=ans_ids, constraint_ids=constraint_ids, pv_official=pv_official, model_dtype=model_dtype, prefix_ok=prefix_ok, fresh_answer=fresh_answer, gen_prompt_inputs=gen_inputs)

def score(adapter, ctx: ItemCtx, pv: torch.Tensor, tau_safe: float=0.0, need_grad: bool=False):
    inp = dict(ctx.inputs)
    inp['pixel_values'] = pv.to(ctx.model_dtype)
    ctxm = torch.enable_grad() if need_grad else torch.no_grad()
    with ctxm:
        L = ctx.L
        n = len(ctx.ans_ids)
        k = len(ctx.constraint_ids)
        rows = adapter.model(**inp, use_cache=False).logits[0, L - 1:L - 1 + k].float()
        logp = torch.log_softmax(rows[:n], dim=-1)
        idx = torch.tensor(ctx.ans_ids, device=rows.device)
        span_logsum = logp.gather(1, idx.view(-1, 1)).sum()
        tokprob = torch.exp(span_logsum / max(n, 1))
        cid = torch.tensor(ctx.constraint_ids, device=rows.device)
        chosen = rows.gather(1, cid.view(-1, 1)).squeeze(1)
        masked = rows.scatter(1, cid.view(-1, 1), torch.full((k, 1), -1000000000.0, device=rows.device, dtype=rows.dtype))
        rival = masked.max(dim=1).values
        margins = chosen - rival
        min_margin = margins.min()
    return {'tokprob': tokprob, 'span_logsum': span_logsum, 'min_margin': min_margin, 'feasible': bool(min_margin.item() >= tau_safe), 'margins': margins}

def fresh_decode(adapter, ctx: ItemCtx, pv: torch.Tensor):
    gen = dict(ctx.gen_prompt_inputs)
    gen['pixel_values'] = pv.to(ctx.model_dtype)
    with torch.no_grad():
        out = adapter._generate(gen, max_new_tokens=int(ctx.row['max_new_tokens']))
    ids = [int(t) for t in out['new_ids'].tolist()]
    text = adapter.parse_answer(adapter.tokenizer.decode(out['new_ids'], skip_special_tokens=True).strip())
    return (text, ids)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def png_bytes_from_raw(raw01: torch.Tensor) -> bytes:
    import io
    a = torch.clamp(torch.round(raw01.detach().float() * 255.0), 0, 255).to(torch.uint8)
    im = Image.fromarray(a.permute(1, 2, 0).cpu().numpy(), mode='RGB')
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return buf.getvalue()
