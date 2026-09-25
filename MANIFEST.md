# Manifest

Every paper table and figure, mapped to the Tier A generator that builds it, the artifact file(s) it reads, and the GPU pipeline stage (Tier B) that produces those artifacts. The machine-readable form is `manifest.json`. Tier A regenerates all of the below from `artifacts/` on a CPU: `python -m src.analysis.build_all`.

| Paper item | Label | Tier A generator | Artifact(s) | Tier B stage |
|---|---|---|---|---|
| Table 1 | `tab:core` | `src.analysis.tables_main:build_core_decomp` | `core_decomp.json`, `label_free.json` | attacks, estimators |
| Table 2 | `tab:ladder` | `src.analysis.tables_main:build_ladder_main` | `ladder_full.json` | verifier_ladder |
| Table 3 | `tab:app-preservation` | `src.analysis.tables_setup:build_answer_preservation` | `answer_preservation.json` | attacks |
| Table 4 | `tab:app-norms` | `src.analysis.tables_setup:build_perturbation_norms` | `perturbation_norms.json` | attacks |
| Table 5 | `tab:app-percept` | `src.analysis.tables_setup:build_perceptibility` | `perceptibility.json` | attacks |
| Table 6 | `tab:app-judge` | `src.analysis.tables_setup:build_judge_validation` | `judge_validation.json` | judge |
| Table 7 | `tab:app-draw` | `src.analysis.tables_setup:build_canonical_draw` | `canonical_draw.json` | data_prep |
| Table 8 | `tab:app-hyper` | `src.analysis.tables_setup:build_hyperparameters` | `configs/hyperparameters.yaml` | config |
| Table 9 | `tab:app-decomp-full` | `src.analysis.tables_longtables:build_decomp_full` | `decomp_full.json` | attacks, estimators |
| Table 10 | `tab:app-subspace` | `src.analysis.tables_mechanisms:build_answer_coupled_subspace` | `answer_coupled_subspace.json` | hidden_states |
| Table 11 | `tab:app-within` | `src.analysis.tables_mechanisms:build_within_answer` | `within_answer.json` | estimators |
| Table 12 | `tab:app-ceiling` | `src.analysis.tables_mechanisms:build_ceiling_floor` | `ceiling_floor.json` | attacks, estimators |
| Table 13 | `tab:app-longanswer` | `src.analysis.tables_mechanisms:build_long_answer_full` | `long_answer_full.json` | attacks |
| Table 14 | `tab:app-certificate` | `src.analysis.tables_longtables:build_certificate` | `certificate.json` | estimators |
| Table 15 | `tab:app-phenomenon` | `src.analysis.tables_longtables:build_phenomenon_full` | `phenomenon_full.json` | attacks |
| Table 16 | `tab:app-labelfree` | `src.analysis.tables_phenomenon:build_label_free` | `label_free.json` | attacks |
| Table 17 | `tab:app-multiseed` | `src.analysis.tables_phenomenon:build_multiseed_attacked` | `multiseed_attacked.json` | attacks |
| Table 18 | `tab:app-multiseed-survivor` | `src.analysis.tables_phenomenon:build_multiseed_survivor` | `multiseed_survivor.json` | attacks |
| Table 19 | `tab:app-kl` | `src.analysis.tables_distribution:build_kl_constraint` | `kl_constraint.json` | attacks |
| Table 20 | `tab:app-kl-surviving` | `src.analysis.tables_distribution:build_kl_surviving` | `kl_surviving.json` | attacks |
| Table 21 | `tab:app-kl-behavioral` | `src.analysis.tables_distribution:build_kl_behavioral` | `kl_behavioral.json` | attacks |
| Table 22 | `tab:app-gemma-sens` | `src.analysis.tables_distribution:build_gemma_sensitivity` | `gemma_sensitivity.json` | attacks |
| Table 23 | `tab:app-gate-harm` | `src.analysis.tables_scope:build_gate_harm` | `gate_harm.json` | estimators |
| Table 24 | `tab:app-gate-robust` | `src.analysis.tables_scope:build_gate_robustness` | `gate_robustness.json` | estimators |
| Table 25 | `tab:app-frontier` | `src.analysis.tables_scope:build_frontier_sweep` | `frontier_sweep.json` | attacks |
| Table 26 | `tab:app-invariance` | `src.analysis.tables_scope:build_attack_invariance` | `attack_invariance.json` | attacks |
| Table 27 | `tab:app-ladder` | `src.analysis.tables_main:build_ladder_full` | `ladder_full.json` | verifier_ladder |
| Table 28 | `tab:app-parity` | `src.analysis.tables_main:build_ladder_parity` | `ladder_parity.json` | verifier_ladder |
| Table 29 | `tab:app-transfer` | `src.analysis.tables_scope:build_cross_model_transfer` | `cross_model_transfer.json` | attacks |
| Table 30 | `tab:app-universal` | `src.analysis.tables_scope:build_universal_perturbation` | `universal_perturbation.json` | attacks |
| Table 31 | `tab:app-defenses` | `src.analysis.tables_scope:build_defenses_partial` | `defenses_partial.json` | attacks |
| Figure 3 | `fig:frontier` | `src.figures.frontier:build_frontier` | `frontier_untrained.json`, `frontier_sweep.json` | attacks, inference |

## Label-free provenance

`label_free.json` (Table 16, and the label-free column of Table 1) is baked by `src.bake.label_free` from `labelfree_clean_sign_all.json`, the seven-readout clean-sign result store. The clean-sign selector keys on each readout's disjoint clean median in `labelfree_disjoint_medians.json`, computed from the out-of-pool clean scores in `labelfree_disjoint/<model>__<set>.jsonl` (104,000 items). The four estimator readouts in those files are produced by the `attacks` stage script `src.attacks.labelfree_disjoint_scores`; the other three readouts reuse the frozen clean scores already stored by the inference and estimator stages.
