from __future__ import annotations
from dataclasses import dataclass
FORMAT_INSTRUCTIONS = {'yesno': 'Answer the question using a single word: yes or no.', 'short': 'Answer the question using a single word or short phrase.', 'open': 'Answer the question directly.'}
GEN_CAPS = {'yesno': 8, 'short': 32, 'open': 256}

@dataclass(frozen=True)
class AnswerSpec:
    format_key: str
    format_instruction: str
    max_new_tokens: int

def answer_spec(dataset: str, answer_type: str) -> AnswerSpec:
    at = (answer_type or '').lower()
    if dataset == 'pope':
        key = 'yesno'
    elif dataset == 'mme_finance':
        key = 'open'
    elif dataset == 'vqav2':
        key = 'yesno' if at in ('yes/no', 'yesno') else 'short'
    elif dataset == 'gqa':
        key = 'yesno' if at in ('verify', 'logical') else 'short'
    elif dataset == 'slake':
        key = 'short'
    else:
        key = 'short'
    return AnswerSpec(key, FORMAT_INSTRUCTIONS[key], GEN_CAPS[key])
