from __future__ import annotations
import hashlib
import torch
WINDOWS = (15, 30, 45, 60)
KEEP = 5
A_SPEC = {'restarts': 1, 'window': 15}
B_SPEC = {'restarts': 2, 'window': 30}
CANDIDATE_UPPER_BOUND = 2 * KEEP * len(WINDOWS) * 3

def u8_sha(u8: torch.Tensor) -> str:
    return hashlib.sha256(u8.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def candidate_id(model, item_id, objective, direction, restart, step, sha) -> tuple:
    return (model, item_id, objective, direction, int(restart), int(step), sha)

class TwoTierWindowedPools:

    def __init__(self, k: int=KEEP, windows=WINDOWS):
        self.k = k
        self.windows = tuple(windows)
        self.all = {w: [] for w in self.windows}
        self.feas = {w: [] for w in self.windows}

    def offer(self, harm, step, score, min_margin, raw01, feasible):
        u8 = None
        for w in self.windows:
            if step > w:
                continue
            for pool, gated in ((self.all[w], False), (self.feas[w], True)):
                if gated and (not feasible):
                    continue
                if len(pool) >= self.k and harm <= pool[-1][0]:
                    continue
                if u8 is None:
                    u8 = torch.clamp(torch.round(raw01.detach() * 255.0), 0, 255).to(torch.uint8).cpu()
                pool.append((harm, step, score, min_margin, u8, feasible))
                pool.sort(key=lambda t: (-t[0], t[1]))
                del pool[self.k:]

    def union(self, restart: int) -> dict:
        out = {}
        for tier, pools in (('all', self.all), ('feasible', self.feas)):
            for w in self.windows:
                for harm, step, score, mm, u8, feas in pools[w]:
                    c = out.get(step)
                    if c is None:
                        out[step] = {'harm': harm, 'step': step, 'score': score, 'min_margin': mm, 'u8': u8, 'tf_feasible': feas, 'restart': restart, 'windows': [w], 'tiers': [tier]}
                    else:
                        if w not in c['windows']:
                            c['windows'].append(w)
                        if tier not in c['tiers']:
                            c['tiers'].append(tier)
        return out

    def subset_ids(self, model, item_id, objective, direction, restart, window) -> set:
        out = set()
        for pools in (self.all, self.feas):
            for harm, step, score, mm, u8, feas in pools[window]:
                out.add(candidate_id(model, item_id, objective, direction, restart, step, u8_sha(u8)))
        return out

    def subset_candidates(self, window) -> list:
        out = {}
        for pools in (self.all, self.feas):
            for t in pools[window]:
                out[t[1]] = t
        return list(out.values())

def reconstruct_from_trajectory(traj, k=KEEP, window=15):
    allp, feasp = ([], [])
    for t in traj:
        if t['step'] > window:
            continue
        for pool, gated in ((allp, False), (feasp, True)):
            if gated and (not t['feasible']):
                continue
            if len(pool) >= k and t['harm'] <= pool[-1]['harm']:
                continue
            pool.append(t)
            pool.sort(key=lambda x: (-x['harm'], x['step']))
            del pool[k:]
    out = {}
    for pool in (allp, feasp):
        for t in pool:
            out[t['step']] = t
    return out

def containment_audit(model, item_id, objective, direction, pools_by_restart, trajectories) -> dict:
    result = {'checks': [], 'pass': True, 'bound_before_dedup': CANDIDATE_UPPER_BOUND}
    for name, spec in (('A_15x1', A_SPEC), ('B_30x2', B_SPEC)):
        want, have = (set(), set())
        for r in range(spec['restarts']):
            if r >= len(pools_by_restart):
                continue
            rec = reconstruct_from_trajectory(trajectories[r], KEEP, spec['window'])
            for step, t in rec.items():
                want.add(candidate_id(model, item_id, objective, direction, r, step, u8_sha(t['u8'])))
            have |= pools_by_restart[r].subset_ids(model, item_id, objective, direction, r, spec['window'])
        missing = sorted(want - have)
        result['checks'].append({'emulates': name, 'restarts': spec['restarts'], 'window': spec['window'], 'n_reconstructed': len(want), 'n_retained': len(have), 'n_missing': len(missing), 'missing_sample': [list(x) for x in missing[:5]], 'contained': len(missing) == 0, 'method': 'set containment on (model, item_id, objective, direction, restart, step, uint8 sha256), not step ranges'})
        if missing:
            result['pass'] = False
    return result

def order_and_select(cands, sgn, clean_u8, official_score_fn, decode_fn, committed_answer):
    for c in cands:
        s, feas, marg, pv = official_score_fn(c['u8'])
        c['official_score'] = s
        c['official_harm'] = sgn * s
        c['official_feasible'] = feas
        c['official_min_margin'] = marg
        c['pv'] = pv
    n_off = len(cands)
    s, feas, marg, pv = official_score_fn(clean_u8)
    n_off += 1
    clean_cand = {'harm': sgn * s, 'step': -1, 'score': s, 'min_margin': marg, 'u8': clean_u8, 'tf_feasible': feas, 'restart': -1, 'windows': [], 'tiers': ['clean_guaranteed'], 'official_score': s, 'official_harm': sgn * s, 'official_feasible': feas, 'official_min_margin': marg, 'pv': pv, 'is_clean': True}
    ordered = sorted(cands + [clean_cand], key=lambda c: -c['official_harm'])
    chosen, n_dec, rejects = (None, 0, [])
    for i, c in enumerate(ordered):
        if c.get('is_clean'):
            text, ids = (committed_answer, None)
            preserved = True
        else:
            text, ids = decode_fn(c['pv'])
            preserved = text == committed_answer
        n_dec += 1
        if preserved:
            chosen = dict(c)
            chosen['decoded_answer'] = text
            chosen['decoded_ids'] = ids
            chosen['order_index'] = i
            break
        rejects.append({'restart': c['restart'], 'step': c['step'], 'official_harm': round(c['official_harm'], 6), 'decoded_answer': text})
    return (chosen, ordered, n_off, n_dec, rejects)

def assert_no_better_later(chosen, ordered, decode_fn, committed_answer, check_all=True) -> dict:
    idx = chosen['order_index']
    violations = []
    checked = 0
    for j, c in enumerate(ordered):
        if j <= idx:
            continue
        if c['official_harm'] <= chosen['official_harm'] + 1e-12:
            break
        if not check_all:
            continue
        text, _ = (committed_answer, None) if c.get('is_clean') else decode_fn(c['pv'])
        checked += 1
        if text == committed_answer:
            violations.append({'step': c['step'], 'restart': c['restart'], 'official_harm': c['official_harm']})
    return {'selected_order_index': idx, 'n_later_checked': checked, 'violations': violations, 'pass': len(violations) == 0}
