from __future__ import annotations
import base64
import io
import os
import re
import time
from typing import Optional
from PIL import Image
MODEL = 'gpt-5.4-mini'

def _encode(image: Image.Image, max_side: int=768, quality: int=85) -> str:
    img = image.convert('RGB')
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

def _clean(s: str) -> Optional[str]:
    s = (s or '').strip().strip(' ."\'`')
    if not s:
        return None
    first = s.splitlines()[0].strip(' ."\'`')
    if first.upper() == 'NONE' or not first:
        return None
    return first

class GPTVision:

    def __init__(self):
        self._client = None
        self.n_calls = 0

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        return self._client

    def _call(self, text: str, image: Optional[Image.Image]=None, max_tokens: int=2000, instructions: Optional[str]=None) -> str:
        content = [{'type': 'input_text', 'text': text}]
        if image is not None:
            content.append({'type': 'input_image', 'image_url': _encode(image)})
        self.n_calls += 1
        last = None
        for attempt in range(5):
            try:
                kw = {'model': MODEL, 'input': [{'role': 'user', 'content': content}], 'max_output_tokens': max_tokens}
                if instructions is not None:
                    kw['instructions'] = instructions
                resp = self.client.responses.create(**kw)
                return resp.output_text or ''
            except Exception as e:
                last = e
                time.sleep(0.5 * 2 ** attempt)
        raise last
    JUDGE_SYSTEM = 'You are an expert answer evaluator. Your task is to determine if a student\'s answer to a question is correct by comparing it to the ground truth answer.\n\n1. Read the question carefully.\n2. Compare the student\'s answer to the ground truth answer.\n3. Consider semantic equivalence, answers that mean the same thing should be considered correct even if worded differently.\n4. Return ONLY "yes" if the answer is correct, or "no" if it is incorrect.\n5. Be lenient with minor variations in wording, capitalization, or punctuation.'

    def judge_answer(self, image, question: str, gold: str, pred: str) -> Optional[bool]:
        user = f"Question: {question}\nGround Truth Answer: {gold}\nStudent Answer: {pred}\nIs the student's answer correct? (yes/no):"
        try:
            out = (self._call(user, image, max_tokens=1500, instructions=self.JUDGE_SYSTEM) or '').strip().lower()
        except Exception:
            return None
        if out.startswith('yes'):
            return True
        if out.startswith('no'):
            return False
        return None

    def resolve_relevant(self, image, question: str, hint: Optional[str]=None):
        hint_line = f'\nA candidate name from annotations (use only if consistent with the image): {hint}' if hint else ''
        prompt = f'You are shown an image and a question about it. Identify the single specific object in the image that one must look at to answer the question. Give a short, concrete noun phrase that an object detector could localize (include a color or position only if it helps disambiguate). Then, after a vertical bar, give a brief location in the image or NONE.\nQuestion: {question}{hint_line}\nReply exactly as: <noun phrase> | <location or NONE>'
        out = self._call(prompt, image)
        name, loc = (out, None)
        if '|' in out:
            name, loc = out.split('|', 1)
        name = _clean(name)
        loc = _clean(loc or '')
        return {'name': name, 'location': loc, 'raw': out.strip()[:160]}

    def pick_competing_menu(self, image, question: str, candidate_names: list[str], relevant_name: Optional[str]):
        if not candidate_names:
            return None
        menu = '; '.join(sorted(set(candidate_names)))
        prompt = f'You are shown an image, a question about it, and a list of objects detected in the image. Pick the single MOST PROMINENT object from the list that is NOT relevant to answering the question (a salient distractor a viewer could ignore and still answer). Do not pick the object the question is about. Reply with the exact object name from the list, or NONE if every listed object is relevant.\nQuestion: {question}\nThe question is about: {relevant_name}\nDetected objects: {menu}\nReply with just the object name.'
        return _clean(self._call(prompt, image))

    def name_competing_free(self, image, question: str, relevant_name: Optional[str]):
        prompt = f'You are shown an image and a question about it. Name one PROMINENT thing in the image that is clearly UNRELATED to answering the question (a salient object or a contentful region such as a wall, sky, or grass patch, not empty periphery). It must be something an object detector could localize and must not be the thing the question is about. Reply with a short noun phrase, or NONE.\nQuestion: {question}\nThe question is about: {relevant_name}\nReply with just the noun phrase.'
        return _clean(self._call(prompt, image))

    def depends_on(self, question: str, competing_name: str) -> Optional[bool]:
        prompt = f'To answer the question "{question}", must a viewer look at the "{competing_name}" in the image? Answer strictly yes or no.'
        out = (self._call(prompt, None, max_tokens=1500) or '').strip().lower()
        if 'yes' in out and 'no' not in out:
            return True
        if 'no' in out:
            return False
        return None
