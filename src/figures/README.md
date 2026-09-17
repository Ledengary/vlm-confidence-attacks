# Figures (Tier A)

This stage regenerates Figure 3, the informativeness-manipulability frontier, from the compact
stores in `artifacts/`. It is CPU only.

## Running

From the repository root:

```
python -m src.figures.frontier
```

This writes the point coordinates to `artifacts/frontier_points.json` and renders
`outputs/figures/frontier.pdf`. It is also called by `python -m src.analysis.build_all`.

## What it plots

- Untrained internal readouts, one point per cell, pooled over the six informative readouts
  (from the witnessed-union profile).
- The strongest adversarially trained probe per cell, and the single-cell penalty sweep.

Inputs: `frontier_untrained.json` and `frontier_sweep.json` in `artifacts/`. The `src/bake/`
scripts produce those stores from the full pipeline outputs.
