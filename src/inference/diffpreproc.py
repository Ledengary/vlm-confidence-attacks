from __future__ import annotations
import math
import torch
from .pil_exact import BICUBIC, BILINEAR, quantize_ste, resize_exact

def _norm(x255: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.as_tensor(mean, dtype=x255.dtype, device=x255.device).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=x255.dtype, device=x255.device).view(-1, 1, 1)
    return (x255 / 255.0 - m) / s

def internvl_geometry(h: int, w: int, image_size: int, max_tiles: int, use_thumbnail: bool):
    ar = w / h
    target_ratios = sorted({(i, j) for n in range(1, max_tiles + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_tiles}, key=lambda x: x[0] * x[1])
    best_diff = float('inf')
    best = (1, 1)
    area = w * h
    for r in target_ratios:
        tar = r[0] / r[1]
        diff = abs(ar - tar)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]):
            best_diff = diff
            best = r
    tw, th = (image_size * best[0], image_size * best[1])
    n_tiles = best[0] * best[1]
    return {'ratio': best, 'tw': tw, 'th': th, 'n_tiles': n_tiles, 'thumbnail': bool(use_thumbnail and n_tiles != 1)}

def internvl_pp(raw01: torch.Tensor, image_size: int, max_tiles: int, use_thumbnail: bool, mean, std) -> torch.Tensor:
    x = quantize_ste(raw01)
    _, h, w = x.shape
    g = internvl_geometry(h, w, image_size, max_tiles, use_thumbnail)
    resized = resize_exact(x, g['th'], g['tw'], BICUBIC)
    cols = g['tw'] // image_size
    tiles = []
    for i in range(g['n_tiles']):
        top = i // cols * image_size
        left = i % cols * image_size
        tiles.append(resized[..., top:top + image_size, left:left + image_size])
    if g['thumbnail']:
        tiles.append(resize_exact(x, image_size, image_size, BICUBIC))
    return _norm(torch.stack(tiles, dim=0), mean, std)

def gemma3_pp(raw01: torch.Tensor, height: int, width: int, resample: int, mean, std) -> torch.Tensor:
    x = quantize_ste(raw01)
    x = resize_exact(x, height, width, resample)
    return _norm(x, mean, std).unsqueeze(0)

def _select_best_resolution(orig_h: int, orig_w: int, possible):
    best_fit = None
    max_eff = 0
    min_waste = float('inf')
    for height, width in possible:
        scale = min(width / orig_w, height / orig_h)
        dw, dh = (int(orig_w * scale), int(orig_h * scale))
        eff = min(dw * dh, orig_w * orig_h)
        waste = width * height - eff
        if eff > max_eff or (eff == max_eff and waste < min_waste):
            max_eff = eff
            min_waste = waste
            best_fit = (height, width)
    return best_fit

def _patch_output_size(orig_h: int, orig_w: int, target_h: int, target_w: int):
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    if scale_w < scale_h:
        return (min(math.ceil(orig_h * scale_w), target_h), target_w)
    return (target_h, min(math.ceil(orig_w * scale_h), target_w))

def llava_ov_geometry(h: int, w: int, grid_pinpoints, patch_size: int):
    best_h, best_w = _select_best_resolution(h, w, grid_pinpoints)
    new_h, new_w = _patch_output_size(h, w, best_h, best_w)
    pad_top, r_y = divmod(best_h - new_h, 2)
    pad_left, r_x = divmod(best_w - new_w, 2)
    return {'best': (best_h, best_w), 'new': (new_h, new_w), 'pad_top': pad_top, 'pad_bottom': pad_top + r_y, 'pad_left': pad_left, 'pad_right': pad_left + r_x, 'n_patches': best_h // patch_size * (best_w // patch_size)}

def llava_ov_pp(raw01: torch.Tensor, size_h: int, size_w: int, grid_pinpoints, resample: int, mean, std) -> torch.Tensor:
    x = quantize_ste(raw01)
    _, h, w = x.shape
    g = llava_ov_geometry(h, w, grid_pinpoints, size_h)
    base = resize_exact(x, size_h, size_w, resample)
    fit = resize_exact(x, g['new'][0], g['new'][1], resample)
    padded = torch.nn.functional.pad(fit, (g['pad_left'], g['pad_right'], g['pad_top'], g['pad_bottom']), mode='constant', value=0.0)
    tiles = [base]
    bh, bw = g['best']
    for top in range(0, bh, size_h):
        for left in range(0, bw, size_w):
            tiles.append(padded[..., top:top + size_h, left:left + size_w])
    return _norm(torch.stack(tiles, dim=0), mean, std).unsqueeze(0)

def _molmo_resize(x255: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    _, in_h, in_w = x255.shape
    if (in_h, in_w) == (out_h, out_w):
        return x255
    soft = torch.nn.functional.interpolate(x255.unsqueeze(0).to(torch.float32), size=(out_h, out_w), mode='bilinear', align_corners=False, antialias=False).squeeze(0)
    hard = torch.clamp(torch.round(soft), 0.0, 255.0)
    return (soft + (hard - soft).detach()).to(x255.dtype)

def _select_tiling(h: int, w: int, patch_size: int, max_num_crops: int):
    tilings = [(i, j) for i in range(1, max_num_crops + 1) for j in range(1, max_num_crops + 1) if i * j <= max_num_crops]
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    req = []
    for i, j in tilings:
        rs = min(i * patch_size / h, j * patch_size / w) if h > 0 and w > 0 else float('inf')
        req.append(rs)
    if all((r < 1 for r in req)):
        ix = max(range(len(req)), key=lambda k: req[k])
    else:
        adj = [10000000000.0 if r < 1.0 else r for r in req]
        ix = min(range(len(adj)), key=lambda k: adj[k])
    return tilings[ix]

def _patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    n, c, h, w = x.shape
    hp, wp = (h // patch, w // patch)
    y = x.permute(0, 2, 3, 1)
    y = y.reshape(n, hp, patch, wp, patch, c)
    y = y.permute(0, 1, 3, 2, 4, 5)
    return y.reshape(n, hp * wp, patch * patch * c)

def molmo2_geometry(h: int, w: int, base_size: int, patch: int, max_crops: int, margins):
    left_margin, right_margin = margins
    total_margin = patch * (left_margin + right_margin)
    crop_patches = base_size // patch
    crop_window_patches = crop_patches - (left_margin + right_margin)
    crop_window_size = crop_window_patches * patch
    tiling = _select_tiling(h - total_margin, w - total_margin, crop_window_size, max_crops)
    src_h = tiling[0] * crop_window_size + total_margin
    src_w = tiling[1] * crop_window_size + total_margin
    return {'tiling': tiling, 'src': (src_h, src_w), 'crop_window_size': crop_window_size, 'n_crops': tiling[0] * tiling[1]}

def molmo2_pp(raw01: torch.Tensor, base_size: int, patch: int, max_crops: int, margins, mean, std) -> torch.Tensor:
    x = quantize_ste(raw01)
    _, h, w = x.shape
    g = molmo2_geometry(h, w, base_size, patch, max_crops, margins)
    base = _norm(_molmo_resize(x, base_size, base_size), mean, std)
    src = _norm(_molmo_resize(x, g['src'][0], g['src'][1]), mean, std)
    crops = []
    cw = g['crop_window_size']
    for i in range(g['tiling'][0]):
        for j in range(g['tiling'][1]):
            crops.append(src[..., i * cw:i * cw + base_size, j * cw:j * cw + base_size])
    allc = torch.stack([base] + crops, dim=0)
    return _patchify(allc, patch)
