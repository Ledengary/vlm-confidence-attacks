from __future__ import annotations
import errno
import fcntl
import json
import os
from dataclasses import dataclass, field
KEY_FIELDS = ('model', 'item_id', 'objective', 'direction', 'config')
REQUIRED_FIELDS = ('model', 'item_id', 'dataset', 'objective', 'direction', 'config', 'clean_objective', 'endpoint_objective', 'endpoint_source', 'used_clean_fallback', 'validity', 'endpoint_scores', 'seconds')
LABEL_FIELDS = ('label', 'group', 'manifest_ref')
MAX_ATTEMPTS = 3

def key_of(rec: dict) -> tuple:
    return tuple((rec.get(f) for f in KEY_FIELDS))

def invalid_reasons(rec: dict) -> list:
    if not isinstance(rec, dict):
        return ['not_an_object']
    if 'ERROR' in rec:
        return ['error_record']
    bad = [f'missing:{f}' for f in REQUIRED_FIELDS if rec.get(f) is None]
    if not any((rec.get(f) is not None for f in LABEL_FIELDS)):
        bad.append('missing:label_or_manifest_ref')
    v = rec.get('validity')
    if isinstance(v, dict):
        for f in ('within_eps', 'in_unit_box', 'finite', 'linf_greylevels'):
            if v.get(f) is None:
                bad.append(f'missing:validity.{f}')
    elif v is not None:
        bad.append('validity_not_an_object')
    es = rec.get('endpoint_scores')
    if es is not None and (not isinstance(es, dict)):
        bad.append('endpoint_scores_not_an_object')
    elif isinstance(es, dict) and (not es):
        bad.append('endpoint_scores_empty')
    return bad

def is_valid_result(rec: dict) -> bool:
    return not invalid_reasons(rec)

@dataclass
class ScanResult:
    done: set = field(default_factory=set)
    records: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    duplicates: dict = field(default_factory=dict)
    conflicting: list = field(default_factory=list)
    invalid: list = field(default_factory=list)
    truncated_files: list = field(default_factory=list)
    n_lines: int = 0
    n_parse_failures: int = 0

    def anomalies(self) -> dict:
        return {'truncated_files': len(self.truncated_files), 'parse_failures': self.n_parse_failures, 'conflicting_keys': len(self.conflicting), 'invalid_records': len(self.invalid), 'error_keys': len(self.errors)}

    def to_json(self):
        return {'n_done': len(self.done), **self.anomalies(), 'n_lines': self.n_lines, 'n_duplicate_keys': len(self.duplicates), 'conflicting_keys_sample': [list(k) for k in self.conflicting[:10]], 'truncated_files_sample': self.truncated_files[:10]}

def scan(paths) -> ScanResult:
    res = ScanResult()
    for p in paths:
        if not os.path.exists(p):
            continue
        raw = open(p, 'rb').read()
        if raw and (not raw.endswith(b'\n')):
            res.truncated_files.append(str(p))
        for line in raw.decode('utf-8', errors='replace').split('\n'):
            line = line.strip()
            if not line:
                continue
            res.n_lines += 1
            try:
                rec = json.loads(line)
            except Exception:
                res.n_parse_failures += 1
                continue
            k = key_of(rec)
            if is_valid_result(rec):
                if k in res.records:
                    res.duplicates[k] = res.duplicates.get(k, 1) + 1
                    prev = res.records[k]
                    if prev.get('endpoint_objective') != rec.get('endpoint_objective') or prev.get('endpoint_source') != rec.get('endpoint_source'):
                        if k not in res.conflicting:
                            res.conflicting.append(k)
                else:
                    res.records[k] = rec
                res.done.add(k)
            elif isinstance(rec, dict) and 'ERROR' in rec:
                res.errors.setdefault(k, []).append(rec)
            else:
                res.invalid.append({'key': [str(x) for x in k], 'reasons': invalid_reasons(rec)[:6]})
    for k in list(res.errors):
        if k in res.done:
            del res.errors[k]
    return res

def pending(all_keys, res: ScanResult, max_attempts: int=MAX_ATTEMPTS) -> tuple:
    todo, exhausted = ([], [])
    for k in all_keys:
        if k in res.done:
            continue
        n = len(res.errors.get(k, []))
        if n >= max_attempts:
            exhausted.append({'key': [str(x) for x in k], 'attempts': n, 'last_error': res.errors[k][-1].get('ERROR', '')[:200]})
        else:
            todo.append(k)
    return (todo, exhausted)

def verify_complete(all_keys, res: ScanResult, max_attempts: int=MAX_ATTEMPTS) -> dict:
    expected = set(all_keys)
    missing = sorted(expected - res.done)
    extra = sorted(res.done - expected)
    _, exhausted = pending(all_keys, res, max_attempts)
    unresolved = [k for k in res.errors if k not in res.done]
    blockers = []
    if missing:
        blockers.append(f'missing_keys:{len(missing)}')
    if extra:
        blockers.append(f'unexpected_keys:{len(extra)}')
    if res.conflicting:
        blockers.append(f'conflicting_duplicates:{len(res.conflicting)}')
    if res.truncated_files:
        blockers.append(f'truncated_files:{len(res.truncated_files)}')
    if res.n_parse_failures:
        blockers.append(f'parse_failures:{res.n_parse_failures}')
    if unresolved:
        blockers.append(f'unresolved_errors:{len(unresolved)}')
    if exhausted:
        blockers.append(f'exhausted_retries:{len(exhausted)}')
    if res.invalid:
        blockers.append(f'invalid_records:{len(res.invalid)}')
    return {'expected': len(expected), 'done': len(res.done), 'missing': len(missing), 'extra': len(extra), 'missing_sample': [[str(x) for x in k] for k in missing[:10]], 'extra_sample': [[str(x) for x in k] for k in extra[:10]], 'duplicate_keys': len(res.duplicates), 'benign_duplicates': len(res.duplicates) - len(res.conflicting), 'blockers': blockers, 'complete': not blockers}

class ShardWriter:

    def __init__(self, path, lock: bool=True):
        self.path = str(path)
        self.lock_path = self.path + '.lock'
        self.lockfd = None
        if lock:
            self.lockfd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 420)
            try:
                fcntl.flock(self.lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(self.lockfd)
                self.lockfd = None
                if e.errno in (errno.EACCES, errno.EAGAIN):
                    raise RuntimeError(f'another writer already holds {self.lock_path}; one writer per shard')
                raise
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 420)

    def write(self, rec: dict):
        buf = (json.dumps(rec) + '\n').encode('utf-8')
        view = memoryview(buf)
        while view:
            n = os.write(self.fd, view)
            if n <= 0:
                raise IOError('short write made no progress')
            view = view[n:]
        os.fsync(self.fd)

    def close(self):
        for fd in (self.fd, self.lockfd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fd = None
        self.lockfd = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

def repair_truncated(path) -> dict:
    raw = open(path, 'rb').read()
    if not raw or raw.endswith(b'\n'):
        return {'path': str(path), 'repaired': False, 'dropped_bytes': 0}
    cut = raw.rfind(b'\n')
    dropped = len(raw) - (cut + 1)
    with open(path, 'wb') as fh:
        fh.write(raw[:cut + 1] if cut >= 0 else b'')
    return {'path': str(path), 'repaired': True, 'dropped_bytes': dropped}
