# Verifier ladder

This stage studies whether an independent verifier model can catch the adversarial images that fool the base model's own confidence. It runs a spanning set of base-verifier-dataset triples and, for each, walks a ladder from a white-box transfer attack down to black-box query attacks and a joint attack that must fool the base answer and the verifier at once.

## Triples and stimuli

- `triples.py`: the base-verifier-dataset triples, with their independence notes (family, backbone, tokenizer, vision stack) and their item and image locations.
- `span12.py`: the twelve-cell span definition and the stimulus directory layout, plus the shared cell and item helpers.
- `regenerate.py`: regenerates the base-model attack stimuli for a cell through the all-channel gate, using the strict estimator bank.

## Verifier and attacks

- `verifier.py`: the verifier readout and the correctness-elicitation prompt used to score an answer as correct or not from the verifier's point of view.
- `hardlabel_attack.py`: the hard-label (decision-only) black-box attack against the verifier.
- `query_attack.py`: the query-based (score-only) black-box attack.
- `c2_joint.py` and `c2_fill.py`: the joint attack that must preserve the base answer and flip the verifier at the same time, and its fill-in runner over the remaining items.

## Scoring, running, analysis

- `span12_score.py`: scores the clean and attacked stimuli through the verifier per triple.
- `span12_runner.py`: the budgeted multi-GPU launcher that schedules the stimulus, scoring and attack jobs across the ladder.
- `span12_analyze.py`: aggregates the per-triple results into the ladder table.
- `data.py`: the cell loader shared with the estimator stores, the image-disjoint split, the answer-string prior, and the AUROC helper.

## Inputs and outputs

Reads the frozen manifests, the base-model estimator stores, the item images, and both models' weights. Writes the per-triple stimuli, verifier scores, attack records and the ladder aggregates under `data/`.

## Running

From the repository root:

```
python -m src.verifier_ladder.span12_score --triple t0_llava_gemma_vqa --gpu 0 --common
python -m src.verifier_ladder.span12_runner --budget-hours 72
python -m src.verifier_ladder.span12_analyze
```

Every stage script carries an argparse help message. The launcher derives the interpreter from the running one, so no interpreter path is hard-coded.
