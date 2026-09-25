from __future__ import annotations
import torch
CORRECTNESS_PROMPT = 'Is this answer correct for the image and question? Answer yes or no.'

def _yes_no_ids(tokenizer):

    def ids_for(words):
        out = set()
        for w in words:
            for enc in (w, ' ' + w):
                t = tokenizer.encode(enc, add_special_tokens=False)
                if t:
                    out.add(t[0])
        return sorted(out)
    return (ids_for(['yes', 'Yes', 'YES']), ids_for(['no', 'No', 'NO']))

class Verifier:

    def __init__(self, adapter):
        self.adapter = adapter
        self.tok = adapter.tokenizer
        self.yes_ids, self.no_ids = _yes_no_ids(self.tok)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError('could not resolve yes/no token ids for the verifier tokenizer')

    def _prompt(self, question: str, base_answer: str) -> str:
        return f'Question: {question}\nProposed answer: {base_answer}\n{CORRECTNESS_PROMPT}'

    @torch.inference_mode()
    def yes_prob(self, image, question: str, base_answer: str) -> dict:
        ad = self.adapter
        inputs = ad.build_inputs(image, self._prompt(question, base_answer))
        inputs = dict(inputs)
        inputs.update(ad._forward_extra(inputs))
        out = ad.model(**inputs, use_cache=False)
        logits = out.logits[0, -1].to(torch.float32)
        probs = torch.softmax(logits, dim=-1)
        p_yes = float(probs[self.yes_ids].sum())
        p_no = float(probs[self.no_ids].sum())
        denom = p_yes + p_no
        return {'p_yes': p_yes, 'p_no': p_no, 'score': p_yes / denom if denom > 0 else 0.5, 'yes_no_mass': denom}

    @torch.inference_mode()
    def score_image_file(self, png_path, question: str, base_answer: str) -> dict:
        from PIL import Image
        return self.yes_prob(Image.open(png_path).convert('RGB'), question, base_answer)
