from __future__ import annotations
import math
from functools import lru_cache
import torch
PRECISION_BITS = 32 - 8 - 2
_ROUND_OFFSET = 1 << PRECISION_BITS - 1
_SCALE = float(1 << PRECISION_BITS)
NEAREST, LANCZOS, BILINEAR, BICUBIC, BOX, HAMMING = (0, 1, 2, 3, 4, 5)

def _bilinear_filter(x: float) -> float:
    x = -x if x < 0.0 else x
    return 1.0 - x if x < 1.0 else 0.0

def _bicubic_filter(x: float) -> float:
    a = -0.5
    x = -x if x < 0.0 else x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0
_FILTERS = {BILINEAR: (_bilinear_filter, 1.0), BICUBIC: (_bicubic_filter, 2.0)}

def _precompute_int_coeffs(in_size: int, out_size: int, resample: int):
    if resample not in _FILTERS:
        raise NotImplementedError(f'resample filter {resample} not implemented')
    filt, support0 = _FILTERS[resample]
    in0, in1 = (0.0, float(in_size))
    scale = (in1 - in0) / out_size
    filterscale = scale if scale >= 1.0 else 1.0
    support = support0 * filterscale
    bounds = []
    kk_int = []
    for xx in range(out_size):
        center = in0 + (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        ww = 0.0
        k = []
        for x in range(xmax):
            w = filt((x + xmin - center + 0.5) * ss)
            k.append(w)
            ww += w
        if ww != 0.0:
            k = [w / ww for w in k]
        ki = [int(-0.5 + w * _SCALE) if w < 0 else int(0.5 + w * _SCALE) for w in k]
        bounds.append((xmin, xmax))
        kk_int.append(ki)
    return (bounds, kk_int)

@lru_cache(maxsize=512)
def _coeff_matrix_cpu(in_size: int, out_size: int, resample: int) -> torch.Tensor:
    bounds, kk = _precompute_int_coeffs(in_size, out_size, resample)
    m = torch.zeros(out_size, in_size, dtype=torch.float64)
    for i, ((xmin, xmax), ki) in enumerate(zip(bounds, kk)):
        for j in range(xmax):
            m[i, xmin + j] = float(ki[j])
    return m
_MATRIX_CACHE: dict = {}

def _coeff_matrix(in_size: int, out_size: int, resample: int, device) -> torch.Tensor:
    key = (in_size, out_size, resample, str(device))
    m = _MATRIX_CACHE.get(key)
    if m is None:
        m = _coeff_matrix_cpu(in_size, out_size, resample).to(device)
        _MATRIX_CACHE[key] = m
    return m

def _shift_clamp_ste(acc: torch.Tensor) -> torch.Tensor:
    soft = acc / _SCALE
    hard = torch.clamp(torch.floor(soft), 0.0, 255.0)
    return soft + (hard - soft).detach()

def resize_exact(x: torch.Tensor, out_h: int, out_w: int, resample: int) -> torch.Tensor:
    if x.dtype != torch.float64:
        x = x.to(torch.float64)
    in_h, in_w = (x.shape[-2], x.shape[-1])
    if in_h == out_h and in_w == out_w:
        return x
    if in_w != out_w:
        wm = _coeff_matrix(in_w, out_w, resample, x.device)
        acc = torch.matmul(x, wm.transpose(0, 1)) + _ROUND_OFFSET
        x = _shift_clamp_ste(acc)
    if in_h != out_h:
        hm = _coeff_matrix(in_h, out_h, resample, x.device)
        acc = torch.matmul(hm, x) + _ROUND_OFFSET
        x = _shift_clamp_ste(acc)
    return x

def quantize_ste(raw01: torch.Tensor) -> torch.Tensor:
    x = raw01 * 255.0
    hard = torch.clamp(torch.round(x), 0.0, 255.0)
    return x + (hard - x).detach()

def crop(x: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor:
    return x[..., top:top + height, left:left + width]
