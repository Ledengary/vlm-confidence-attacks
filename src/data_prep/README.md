# Data preparation

This stage turns the raw source datasets into the frozen, balanced item pools that every later stage reads. The released path reconstructs the pools from committed seed-23 identifiers rather than re-sampling, so the exact 12-cell matrix (four models by three benchmarks, 1000 items per cell at 500 correct and 500 incorrect) is fixed and reproducible.

All randomness is seeded with 23. The canonical assembler is read-only with respect to the frozen pools and only writes a per-model manifest.

## Committed pool identifiers

`artifacts/pools/<model>__pool.json` holds the frozen seed-23 pool for each model: the list of item identifiers (with per-item image hashes) that define the 1000-item balanced pool for that model across the three benchmarks. These id lists are the source of truth for the item set; the InternVL pool reuses its earlier draw and is included here in the same form. Downstream stages key on these identifiers, so the pools never need to be re-sampled.

## Dataset adapters

`datasets/` holds a small pluggable interface (`base.py`) and one adapter per dataset behind `datasets/registry.py`:

- `gqa_adapter.py` (with `gqa.py`): GQA balanced splits, scene-graph boxes, structural question type as the stratification key.
- `vqav2_adapter.py`: VQAv2 validation, official answer_type as the stratification key, COCO val2014 images.
- `pope_adapter.py`: POPE test items across the three sampling regimes, yes/no answers.

`coco_boxes.py` builds and caches a COCO image-id to ground-truth-box index used by the VQAv2 and POPE adapters. Each adapter yields lightweight `DatasetItem` records (metadata only, no pixels) and fetches an item's image on demand. Images are read from the local Hugging Face cache (`HF_HOME`) and are not shipped here.

## Inputs

- The source datasets in the Hugging Face cache (GQA, VQAv2, POPE) and the COCO detection parquet used by `coco_boxes.py`.
- The committed pool identifiers under `artifacts/pools/`.

## Outputs

- `canonical.py` verifies each model's pool against the committed identifiers and writes the per-model item manifest and a summary under `data/iclr2027/manifest/`, which the attack and estimator stages consume.
- `final_freeze.py` and `finalize_freeze.py` document the original stratified over-draw and the surviving-item freeze that produced the committed identifiers; they write draws and id manifests under `data/` when re-run.

## Long-answer cell (A-OKVQA)

The long-answer robustness result (Table 13) uses a separate cell built from A-OKVQA (Schwenk et al., ECCV 2022), whose answers are long enough that byte-identity is a real constraint. It is independent of the GQA, VQAv2, and POPE cells above and uses its own probes.

- `artifacts/long_answer_item_ids.json` is the frozen 1000-item id list (`aokvqa_<question_id>`), the source of truth for the cell's item set.
- `aokvqa.py` downloads A-OKVQA and reconstructs the item pool from that id list rather than resampling: `python -m src.data_prep.aokvqa --out-dir data/long_answer`. It writes `aokvqa_items.json` and one PNG per item under `images/`.
- `src/estimators/long_answer_probes.py` trains the per-cell P(IK) and SAPLMA probes on this cell, image-disjoint (items grouped by image content hash, seed 23, 60/40 train/eval): `python -m src.estimators.long_answer_probes --model <model> --gpu 0`, writing `data/long_answer/<model>/probes.pt` and `split.json`. These probes are distinct from the core-matrix probes and are the ones Table 13 uses.

## Running

From the repository root, each script runs as a module.

```
python -m src.data_prep.canonical
```

The original draw used a region-sourcing step (object detection and segmentation) that is not part of this release. It is not needed to reconstruct the pools: the committed identifiers in `artifacts/pools/` fully determine the item set, and `canonical.py` reconstructs each pool from them.
