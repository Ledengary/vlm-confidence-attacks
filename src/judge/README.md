# Correctness judge

This stage assigns a binary correctness label to every generated answer. Labels come from a
vision judge, not from string matching. The judge is shown the image as a separate multimodal
input alongside the question, the ground-truth answer, and the student answer, and returns yes
or no. A normalized string match is computed as well, but only as the comparison arm for the
label-agreement analysis; it is used as the label only when a judge call errors, so no item is
left unlabeled.

`gpt_vision.py` holds the `GPTVision` client. It uses `gpt-5.4-mini` and reads the API key from
the `OPENAI_API_KEY` environment variable. Requests downscale the image to bound token cost and
retry with exponential backoff on transient errors. `pipe_judge.py` is the driver.

## Inputs

- The per-item answer records written by the inference stage under
  `data/pipeline/<model>/<set>/shard_*.jsonl` (item id, question, gold answer, student answer).
- The item images, fetched through the dataset adapters in `src/data_prep/datasets/`, and the
  item records under `data/draws/`.
- `OPENAI_API_KEY` in the environment.

## Outputs

`data/pipeline/<model>/judged.jsonl`, one row per item with the correctness label, the string
match label, and a flag recording whether the judge or the string-match fallback produced the
label.

## Running

From the repository root:

```
export OPENAI_API_KEY=...
python -m src.judge.pipe_judge --model internvl3_5_2b --workers 100
```

The driver is thread-pooled over the API and resumable: items already present in
`judged.jsonl` are skipped on a rerun.
