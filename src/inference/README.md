# Inference

This stage runs the vision language models. It has three layers: the per-family model
adapters, one fixed forward configuration shared by every measurement, and the clean
generation pass that produces the answer and hidden-state store.

Image and text formatting always come from each model's own processor and chat template (or,
for the legacy InternVL chat model, its own conversation template). Layer counts, hidden
sizes and input formats are never hard-coded; they are read from the loaded model at runtime.

## Model adapters (`models/`)

`models/base.py` defines the adapter contract every script codes against. `ProcessorAdapter`
covers any model that ships an `AutoProcessor` with a chat template (Molmo2, Gemma3,
LLaVA-OneVision); InternVL's legacy chat model has its own adapter in `internvl.py`. Build
adapters only through `models.factory.build_adapter(key)`.

The contract exposes greedy generation with per-token answer-span probabilities, a verbalized
confidence readout, fixed-answer teacher forcing under one or many images
(`score_answer_ids` / `score_answer_ids_batch`), and the P(IK) and SAPLMA hidden states from a
single forward (`answer_hidden_pik_saplma`).

The four models are `internvl3_5_2b`, `molmo2_4b`, `gemma3_12b`, and `llava_onevision_7b`,
registered in `src/config.py`.

## Fixed forward configuration (`engine.py`, `families.py`)

`engine.py` fixes one forward configuration (bfloat16, one CUDA device per worker, InternVL
eager attention and the others sdpa, batch one, no cache in scoring, greedy generation) and
uses it for clean generation, teacher forcing, and fresh decoding. The clean answer token ids
are recovered by re-running the greedy decode rather than by retokenizing a stored string.
Teacher forcing runs over prompt plus the non-special answer span.

`families.py` binds each adapter to a differentiable preprocessing path built from the loaded
model config or official processor. `diffpreproc.py` and `pil_exact.py` implement that path: a
bit-exact differentiable port of the 8-bit image resampler and each model's full preprocessing
from raw RGB in [0, 1].

`answer_format.py` resolves the answer-type-keyed format instruction and generation cap per
dataset and answer type.

## Inputs

- The frozen id-manifests under `manifests/final/` and the item records under `data/draws/`.
- The item images, fetched through the dataset adapters in `src/data_prep/datasets/`.
- The model weights, downloaded to the local Hugging Face cache on first load.

## Outputs

`pipe_inference.py` writes, per model and set, one shard file pair under
`data/pipeline/<model>/<set>/`: a `.npz` with the SAPLMA and P(IK) hidden states and a
`.jsonl` with the answer text, per-answer-token probabilities, the token-probability geometric
mean, and the token count. Correctness is not decided here; that is the judge stage.

## Running

From the repository root, one worker of the inference stage:

```
python -m src.inference.pipe_inference --model internvl3_5_2b --set gqa_val_eval --gpu 0 --shard 0 --num-shards 28
```

Each shard is resumable: a shard whose output files already exist is skipped. Every stage
script has an argparse help message.
