# Analysis and table generation (Tier A)

This stage regenerates every table in the paper from the compact stores in `artifacts/`. It is
CPU only and needs no model weights, no GPU, and no API keys.

## Running

From the repository root:

```
python -m src.analysis.build_all
```

This reads `artifacts/` and writes each table to `outputs/tables/` (main text) and
`outputs/tables/appendices/` (appendix), and calls the figure generator for Figure 3.

## Layout

- `build_all.py` is the entry point; it imports the per-group generator modules and runs every
  builder.
- `tables_main.py`, `tables_setup.py`, `tables_mechanisms.py`, `tables_longtables.py`,
  `tables_phenomenon.py`, `tables_distribution.py`, `tables_scope.py`: one generator function per
  table, grouped by appendix section.
- `templates.py`: the caption and footer text for the tables that carry a caption inside the
  file; every number in a caption or footer is computed from the store at generation time.
- `latexlib.py`: shared helpers (artifact loading, model and set names, number formatting,
  output writing).

## Inputs and outputs

- Inputs: the JSON and YAML stores in `artifacts/` (and `configs/hyperparameters.yaml`).
- Outputs: the `.tex` table bodies under `outputs/`. `MANIFEST.md` maps each table to its
  generator and the artifact it reads.

The `src/bake/` scripts produce the `artifacts/` stores from the full pipeline outputs; a
reviewer does not run them, they run once when the release is assembled.
