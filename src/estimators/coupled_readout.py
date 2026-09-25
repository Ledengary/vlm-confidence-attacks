from __future__ import annotations
import json
import numpy as np
import torch
from src.config import DATA_DIR
from src.inference.models.internvl import IMAGENET_STD
from src.estimators.train_probe import SAPLMANet
from src.hidden_states.common import _score_inputs
__all__ = ['image_std_mean', 'Subjects3', '_answer_preserved', '_run_pgd', '_margin', '_greedy', '_score_inputs']
PIPE = DATA_DIR / 'pipeline'
STEPS = 15
TAU_SAFE = 1.0

def image_std_mean(adapter):
    try:
        ip = getattr(adapter.processor, 'image_processor', None)
        std = getattr(ip, 'image_std', None)
        if std is not None:
            return float(np.mean(np.asarray(std, dtype=np.float64)))
    except Exception:
        pass
    return float(np.mean(IMAGENET_STD))

class Subjects3:

    def __init__(self, model, dev):
        meta = json.load(open(PIPE / model / 'probes_meta.json'))['channels']
        self.dim = int(meta['pik']['input_dim'])
        self.pik_sw = bool(meta['pik']['swapped'])
        self.sap_sw = bool(meta['saplma']['swapped'])
        self.pik = SAPLMANet(self.dim).to(dev)
        self.pik.load_state_dict(torch.load(PIPE / model / 'probe_weights' / 'pik.pth', map_location=dev))
        self.pik.eval()
        self.sap = SAPLMANet(self.dim).to(dev)
        self.sap.load_state_dict(torch.load(PIPE / model / 'probe_weights' / 'saplma.pth', map_location=dev))
        self.sap.eval()

    def compute(self, logits, L, ans_ids, hs_last):
        h_pik = hs_last[L - 1].float()
        h_sap = hs_last[-1].float()
        probs = [torch.softmax(logits[L - 1 + i], dim=-1)[t] for i, t in enumerate(ans_ids)]
        tok = torch.exp(torch.stack([torch.log(p.clamp_min(1e-09)) for p in probs]).mean())
        pp = torch.sigmoid(self.pik(h_pik)).squeeze()
        pik = 1 - pp if self.pik_sw else pp
        sp = torch.sigmoid(self.sap(h_sap)).squeeze()
        sap = 1 - sp if self.sap_sw else sp
        return {'tokprob': tok, 'pik': pik, 'saplma': sap}

def _answer_preserved(logits, L, ans_ids, tau_safe):
    ok = True
    for i, tid in enumerate(ans_ids):
        row = logits[L - 1 + i]
        clean = row[tid]
        rival = row.clone()
        rival[tid] = -1000000000.0
        if float(clean - rival.max()) < tau_safe:
            ok = False
            break
    return ok

def _run_pgd(adapter, subj, inputs2, L, ans_ids, target, sgn, pv0, md, eps_pv, alpha):
    with torch.no_grad():
        o0 = adapter.model(**{**inputs2, 'pixel_values': pv0.to(md)}, output_hidden_states=True, use_cache=False)
        base = float(subj.compute(o0.logits[0].float(), L, ans_ids, o0.hidden_states[-1][0])[target])
    best_val = base
    best_mv = 0.0
    found = False
    best_delta = torch.zeros_like(pv0)
    ok = 0
    tot = 0
    delta = torch.zeros_like(pv0)
    for _ in range(STEPS):
        delta.requires_grad_(True)
        out = adapter.model(**{**inputs2, 'pixel_values': (pv0 + delta).to(md)}, output_hidden_states=True, use_cache=False)
        logits = out.logits[0].float()
        hs = out.hidden_states[-1][0]
        sc = subj.compute(logits, L, ans_ids, hs)
        grad = torch.autograd.grad(sgn * sc[target], delta)[0]
        with torch.no_grad():
            tot += 1
            if _answer_preserved(logits, L, ans_ids, TAU_SAFE):
                ok += 1
                mv = (float(sc[target]) - base) * sgn
                if mv > best_mv:
                    best_mv = mv
                    best_val = float(sc[target])
                    found = True
                    best_delta = delta.detach().clone()
            delta = (delta + alpha * grad.sign()).clamp(-eps_pv, eps_pv).detach()
    return (best_delta, base, best_val, found, ok / max(tot, 1))

def _margin(adapter, inputs2, adv_pv, L, ans_ids, md):
    with torch.no_grad():
        logits = adapter.model(**{**inputs2, 'pixel_values': adv_pv.to(md)}, use_cache=False).logits[0].float()
    ms = []
    for i, tid in enumerate(ans_ids):
        row = logits[L - 1 + i]
        clean = float(row[tid])
        r = row.clone()
        r[tid] = -1000000000.0
        ms.append(clean - float(r.max()))
    return float(min(ms)) if ms else None

def _greedy(adapter, img, question, spec, adv_pv, md):
    gen = dict(adapter.build_inputs(img, adapter.answer_prompt(question, spec.format_instruction)))
    gen['pixel_values'] = adv_pv.to(md)
    out = adapter._generate(gen, max_new_tokens=spec.max_new_tokens)
    raw = adapter.tokenizer.decode(out['new_ids'], skip_special_tokens=True).strip()
    return adapter.parse_answer(raw)
