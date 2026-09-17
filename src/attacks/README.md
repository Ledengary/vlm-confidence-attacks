# Attacks

This stage runs the valid-input, answer-fixed adversarial attacks against the confidence
estimators. The image is optimized in raw RGB in [0,1], every update projects onto the exact
intersection of the epsilon-box and the unit box, and the model's full preprocessing sits
differentiably downstream. The attacked answer is held fixed: argmax preservation is required at
every generated position including termination, and every retained candidate is re-verified by a
fresh greedy decode that must be byte-identical to the clean answer. No item is ever dropped; an
item with no accepted candidate keeps its clean image and its clean score.

## Core token attack

- `attack.py`: one direction of the label-free bidirectional attack on the token-probability
  channel. Sign-gradient ascent on the answer-span log-likelihood, a checkpointed step schedule,
  multiple restarts, a feasible-candidate pool, and a validity ledger. `run_matrix.py` drives the
  full canonical matrix, both the primary bidirectional run and the repaired-legacy comparison run.
- `official_pv.py`: applies each model's own processor to a uint8 adversarial image, so every
  reported quantity is measured through the deployed preprocessing rather than the search path.

## All-channel gate

- `readout_bank.py` and `objectives.py` live in the estimator stage; this stage consumes them.
- `gate_attack.py`: the single-objective gate step, inheriting the raw-RGB geometry and the
  preservation constraint from the core attack.
- `gate_v12.py`: the full-matrix all-channel gate that attacks each estimator and cross-scores the
  others from the same forward. It uses the candidate `selection.py` pools and the crash-safe
  `store.py` shard writer, and reads the estimator bank through `canonical_bank.py`, whose strict
  variant `strict_bank.py` pins the feature column order.
- `panel.py` builds and loads the evaluation panel of cells.

## Constrained and multi-seed variants

- `kl_gate.py` and `run_cell.py`: the KL-bounded variant, which wraps the gate so each update is
  additionally constrained by a bound on the output-distribution shift, and its per-cell runner.
- `run_cell_ms.py`, `ms_runner.py`, `aggregate_ms.py`: the multi-seed variant that reruns the core
  matrix under several seeds and aggregates the results. `definitions.py` holds the label-free
  descriptive readouts (AUROC with tie half-credit and the witnessed worst-case oracle selector).

## Inputs and outputs

Reads the frozen manifests, the trained estimator weights and banks, and the item images. Writes,
per model and cell, the per-item attack records (clean and endpoint scores, displacement, validity,
provenance) and the accepted adversarial PNGs under `data/`.

## Running

From the repository root, one worker per GPU, sharded and resumable:

```
python -m src.attacks.run_matrix --run primary --model internvl3_5_2b --gpu 0 --shard 0 --num-shards 1 --config C_60x3
python -m src.attacks.gate_v12 --help
python -m src.attacks.run_cell --help
python -m src.attacks.ms_runner --budget-hours 48
```

Every stage script carries an argparse help message. The subprocess launchers derive the
interpreter from the running one, so no interpreter path is hard-coded.
