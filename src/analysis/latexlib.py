import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(REPO_ROOT, "artifacts")
OUTPUTS = os.path.join(REPO_ROOT, "outputs")

MODEL_ORDER = ["internvl3_5_2b", "molmo2_4b", "llava_onevision_7b", "gemma3_12b"]
SET_ORDER = ["gqa_val_eval", "vqav2_ood", "pope_ood"]

MODEL_LONG = {
    "internvl3_5_2b": "InternVL3.5-2B",
    "molmo2_4b": "Molmo2-4B",
    "llava_onevision_7b": "LLaVA-OV-7B",
    "gemma3_12b": "Gemma-3-12B",
}
MODEL_SHORT = {
    "internvl3_5_2b": "InternVL",
    "molmo2_4b": "Molmo",
    "llava_onevision_7b": "LLaVA",
    "gemma3_12b": "Gemma",
}
SET_SHORT = {"gqa_val_eval": "GQA", "vqav2_ood": "VQAv2", "pope_ood": "POPE"}


def load_artifact(name):
    with open(os.path.join(ARTIFACTS, name)) as f:
        return json.load(f)


def write_output(relpath, lines):
    path = os.path.join(OUTPUTS, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return path


def fnum(x, nd):
    return f"{x:.{nd}f}"
