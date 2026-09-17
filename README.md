# Vision-Language Model Confidence Is Not a Property of the Answer

Code and released artifacts for the paper. The paper studies answer-preserving adversarial
attacks on vision-language model confidence: an imperceptible change to the image holds the
model's answer byte-for-byte identical while driving every model-internal confidence readout
from informative to inverted, below chance. It covers four models (InternVL3.5-2B, Molmo2-4B,
LLaVA-OneVision-7B, Gemma-3-12B) on three benchmarks (GQA, VQAv2, POPE), seven confidence
readouts, an exact decomposition that explains the one profile that resists, and a verifier
reachability ladder.

There are two ways to use this repository. Tier A regenerates every table and figure in the
paper on a CPU in minutes from the small released artifact stores, with no model weights, no
GPU, and no API keys. Tier B is the full pipeline from images and checkpoints; it is provided
in full and is smoke-testable on a single cell.

## Tier A: regenerate the tables and figures (CPU, minutes)

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m src.analysis.build_all
```

This reads `artifacts/` and writes every table `.tex` and the frontier figure to `outputs/`.
It takes a few minutes and needs only numpy, scipy, pandas, and matplotlib. `MANIFEST.md` maps
each table and figure to the script that builds it and the artifact it reads.

## Tier B: the full pipeline (GPU)

Tier B runs preprocessing, inference, hidden-state extraction, attacks, probe and estimator
training, the verifier ladder, and judge labeling, writing the stores that Tier A consumes.
The reference stack is in `models.md`. See each stage README under `src/` for inputs, outputs,
and how to run a stage. We do not rerun the full pipeline for a release; `tests/` contains a
one-cell smoke test that executes every stage on a handful of items and checks that the outputs
are in the format Tier A reads, and that re-attacking those items reproduces the frozen
endpoints bit-for-bit.

The smoke test runs from a fresh clone using the curated InternVL3.5-2B weights in `weights/`
and the eight-item inputs in `tests/smoke_data/` (no other stores or downloads beyond the
model checkpoint from the Hub):

```
pip install -r requirements-full.txt
python tests/smoke_tier_b.py --gpu 0
```

It loads the model and the seven-readout estimator bank, runs clean inference, hidden-state
extraction, the attack, and all seven readouts on the eight InternVL3.5-2B GQA items, and checks
each endpoint's sha256 against the frozen values in `tests/smoke_data/frozen_sha.json`. InternVL
uses eager attention and is bit-deterministic, so all eight endpoints match. It takes a few
minutes on one GPU and prints per-stage runtime. The other three models use SDPA attention, whose
reductions are not bit-deterministic, so their endpoints are valid but not bit-reproducible; the
curated weights here therefore cover the deterministic model.

## Layout

```
src/data_prep/       download and preprocess GQA, VQAv2, POPE; reconstruct the seed-23 pools
src/inference/       clean answer generation with per-token answer-span probabilities
src/hidden_states/   hidden-state extraction for the probe readouts
src/attacks/         the answer-preserving valid-input attack and its variants
src/estimators/      the seven confidence readouts and their trainers
src/verifier_ladder/ the spanning-twelve reachability ladder
src/judge/           the image-aware correctness judge and its prompt
src/analysis/        table generators and the Tier A entry point build_all.py
src/figures/         the frontier figure generator
src/bake/            builders that reduce the large pipeline outputs to the compact artifacts/
artifacts/           compact stores that Tier A reads
outputs/             prebuilt tables and figure, as a reference to diff against
weights/             frozen probe and estimator weights for the smoke test
configs/             models, cells, and hyperparameters
tests/               Tier A and per-stage smoke tests
```

## Requirements and timing

Tier A: Python 3.11, `requirements.txt` (numpy, scipy, pandas, matplotlib, PyYAML). Runs on a
CPU in a few minutes. Tier B: the stack in `models.md`, H200-class GPUs, bfloat16. The one-cell
smoke test runs in minutes on a single GPU.

## Citation

A citation will be added on publication.

## License

MIT, see `LICENSE`.
