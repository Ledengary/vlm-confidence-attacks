# Estimators

This stage defines the confidence channels and trains the estimators that read them. It sits on top of the hidden-state stage: it consumes the cached hidden states and CCPS features and produces the per-item confidence scores and the trained probe weights that the attack stage optimizes against.

## Channel definitions

`channels.py` fixes the channel set and the answer-type-keyed prompt convention and generation caps, each cited to prior work: the token-probability channel (length-normalized sequence likelihood, primary is the geometric mean of the per-token answer-span probabilities), the verbalized channel (Tian et al. 2023 second-stage elicitation applied to our own answer), and the SAPLMA internal-state probe (Azaria and Mitchell). This module measures nothing; it defines.

## Trained estimators

- `train_probe.py`: the paper-faithful SAPLMA probe (256-128-64 MLP, class-balanced BCE, early stop on the validation composite, minority-class label swap). Trains on the probe-train pool, selects on the probe-val pool, scores the evaluation sets.
- `pipe_03_probes.py`: trains two probes of identical style on the canonical pipeline hidden states with the vision-judged labels, the SAPLMA probe on the answer-final-token hidden and the P(IK) probe on the prompt-final-token hidden.
- `ccps_train.py`: the CCPS two-stage training and prediction (a contrastive convolutional encoder, then a jointly fine-tuned classifier), used as published, on the feature tables from the hidden-state stage.
- `ccpsd.py`: CCPS-D, a differentiable surrogate of CCPS treated as its own estimator, plus its answer-fixed adaptive attack. The feature set is the cleanly differentiable subset of the CCPS features so the whole readout stays in the autograd graph from the image.
- `coupled_extract.py`: the extraction pass for the answer-coupled estimator, materializing the clean and answer-fixed-attacked hidden states at both probe loci for every split.
- `channels_readout.py`: the generation readout of the verbalized and self-consistency channels on the clean and stored adversarial input.

## Attack-time estimator bank

`readout_bank.py` is the attack-time bank that exposes all seven estimators (token-probability, P(IK), SAPLMA, CCPS-D, CCPS, FULL, COUPLED) behind one interface, with `objectives.py` giving the five differentiable attacked objectives and the shared answer-preservation constraint from a single forward. `coupled_readout.py` holds the differentiable three-estimator readout, the fixed-answer projected gradient step, and the answer-margin and fresh-decode helpers used by the coupled and CCPS-D passes.

## Metrics helpers

`metrics_lite.py` and `ccps_metrics.py` provide dependency-light AUROC, average precision, ECE and Brier so the estimators are scored on the same footing.

## Inputs and outputs

Reads the hidden-state and CCPS-feature stores under `data/` and the vision-judged labels. Writes the trained weights, the per-item confidence scores, and the per-estimator metrics under `data/`.

## Running

From the repository root:

```
python -m src.estimators.pipe_03_probes --model internvl3_5_2b --device cuda:0
python -m src.estimators.ccps_train --model gemma3_12b --mode train --gpu 0
python -m src.estimators.coupled_extract --model molmo2_4b --split gqa_val_eval --gpu 0 --shard 0 --num-shards 8
```

Every stage script carries an argparse help message.
