# Models and software stack

## Checkpoints

The four vision-language models are used exactly as released, by their Hugging Face identifiers. No weights are redistributed here; they are downloaded from the Hub at run time.

| Internal id | Hugging Face repository | Family | Notes |
|---|---|---|---|
| internvl3_5_2b | `OpenGVLab/InternVL3_5-2B-Instruct` | internvl | requires `trust_remote_code=True` |
| molmo2_4b | `allenai/Molmo2-4B` | molmo | requires `trust_remote_code=True` |
| llava_onevision_7b | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | llava_onevision | LlavaOnevision (Qwen2-7B LM + SigLIP) |
| gemma3_12b | `google/gemma-3-12b-it` | gemma3 | |

Image and text formatting come from each model's own processor and chat template. Layer counts, hidden sizes, and input formats are never hard-coded.

## Pinned software stack (for the GPU pipeline)

Python 3.11. Install the PyTorch stack from the CUDA 12.8 wheels, then the rest:

```
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-full.txt
```

Key versions: torch 2.9.0 (cu128), torchvision 0.24.0, transformers 4.57.3, accelerate 1.10.1, tokenizers 0.22.2, safetensors 0.8.0, huggingface-hub 0.36.2, numpy 1.26.4, pillow 12.2.0, opencv-python-headless 4.11.0.86, scipy 1.17.1, datasets 5.0.0, pandas 3.0.3, matplotlib 3.11.0, openai 2.44.0, triton 3.5.0. The full transitive lock is in `requirements-lock.txt`.

Reference hardware: a single node with H200 GPUs, bfloat16 throughout. One forward configuration is fixed for all models: bfloat16, InternVL under eager attention and the others under SDPA, batch size 1, `use_cache=False`.

## Correctness judge

The correctness label is assigned by an image-aware LLM judge (`gpt-5.4-mini`) that sees the image, question, ground truth, and student answer and returns yes or no. The judge prompt and script are in `src/judge/`. The stored labels are shipped, so the CPU table-regeneration path needs no key. Running the judge in the GPU pipeline requires an API key in the environment variable `OPENAI_API_KEY`.
