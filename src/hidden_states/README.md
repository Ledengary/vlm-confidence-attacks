# Hidden states

This stage materializes the internal-state features that the confidence estimators are trained
and scored on. It runs the models forward over image plus question plus answer and writes the
hidden states to disk, so the estimator stage never re-extracts. It also holds the CCPS
perturbation-stability feature computation, which is a hidden-state readout rather than an
estimator in its own right.

Every locus is read from the loaded model at runtime. Nothing here hard-codes a layer count or a
hidden size, and image and text formatting come from each model's own processor and chat template
through the adapters in `src/inference/`.

## What it produces

- `extract_saplma.py`: for each model and set, forwards the original image plus question plus
  answer and saves the last-layer last-token hidden state (the SAPLMA locus, Azaria and Mitchell)
  with the correctness label. The three evaluation sets reuse a cached answer and label; the two
  probe pools generate the answer here and label it with the string-match comparison arm.
- `pipe_h1_extract.py`: the clean precache. Per item it stores the SAPLMA (answer-final-token) and
  P(IK) (prompt-final-token, Kadavath et al.) hidden states from a single forward, plus the
  token-probability, verbalized, SAPLMA-probe and P(IK)-probe confidences.
- `pipe_layers_extract.py`: the all-layer sweep. Per item it caches the hidden state at every layer
  at both loci, so the layer choice can be studied as a sensitivity check rather than searched.
- `ccps_features.py`: the CCPS stability features (Contrastive Confidence from Perturbation
  Stability). The feature-computation functions are used as published, on the analytical-gradient
  path with `pei_radius=20`, `pei_steps=5`, and the 75-feature vector. Only the model input and
  output are bound to the adapters here. It emits one feature row per answer token, grouped by item
  into a per split table, for both the clean and the stored adversarial image.
- `common.py`: shared helpers. Downscaling to the scoring resolution, the teacher-forcing input
  builder that fixes the prompt-plus-answer sequence so only `pixel_values` varies, the
  pixel-values delta injector, and the string-match answer scorer.

## Inputs

- The frozen id-manifests under `manifests/final/` and the item records under `data/draws/`.
- The item images through the dataset adapters in `src/data_prep/datasets/`.
- For the adversarial CCPS features, the stored per-item clean PNG and pixel-values delta.

## Outputs

Hidden-state stores under `data/` (npz for the hidden tensors, jsonl for the per-item metadata),
and the CCPS feature tables. Correctness is not decided here.

## Running

From the repository root, one worker of a stage, sharded and resumable:

```
python -m src.hidden_states.extract_saplma --model internvl3_5_2b --set gqa_val_eval --gpu 0 --shard 0 --num-shards 28
python -m src.hidden_states.pipe_h1_extract --model gemma3_12b --set gqa_val --gpu 0 --shard 0 --num-shards 8
python -m src.hidden_states.ccps_features --model molmo2_4b --split test --image-mode clean --gpu 0 --shard 0 --num-shards 8
```

Every stage script carries an argparse help message. A shard whose output files already exist is
skipped.
