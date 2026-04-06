# Paper-to-Evidence Map

This file maps the main paper claims, tables, and figures to the scripts and public evidence retained in this repository.

## 1. Method Definition

Paper claim:

- WPHeadMixer extends WPMixer with modular prediction heads over a shared wavelet-patch backbone.
- The released implementation supports linear, MLP, KAN, and hybrid heads.

Repository support:

- entry point and CLI arguments: [code/wpheadmixer/run_LTF.py](code/wpheadmixer/run_LTF.py)
- model wrapper definitions: [code/wpheadmixer/models/model.py](code/wpheadmixer/models/model.py), [code/wpheadmixer/models/model_kan.py](code/wpheadmixer/models/model_kan.py)
- backbone and head routing: [code/wpheadmixer/models/wavelet_patch_mixer.py](code/wpheadmixer/models/wavelet_patch_mixer.py)
- wavelet decomposition and reconstruction: [code/wpheadmixer/models/decomposition.py](code/wpheadmixer/models/decomposition.py)
- KAN layer implementation: [code/wpheadmixer/kan.py](code/wpheadmixer/kan.py)

## 2. Main Benchmark Table

Paper role:

- supports the main linear-head benchmark table discussed in the manuscript

Public evidence:

- [evidence/main_results/linear_main_results_summary.md](evidence/main_results/linear_main_results_summary.md)

Regeneration script:

- [code/wpheadmixer/scripts/run_linear_main_body.py](code/wpheadmixer/scripts/run_linear_main_body.py)

## 3. Compact Multi-Seed Verification

Paper role:

- supports the claim that head ranking is not based on a single lucky seed

Public evidence:

- [evidence/multi_seed/small_multi_seed_verification_summary.md](evidence/multi_seed/small_multi_seed_verification_summary.md)

Regeneration script:

- [code/wpheadmixer/scripts/run_small_seed_verification.py](code/wpheadmixer/scripts/run_small_seed_verification.py)

Coverage:

- datasets: `ETTh1`, `ETTm1`, `Weather`
- horizons: `96`, `720`
- heads: `linear`, `kan`
- seeds: `42`, `43`, `44`

## 4. Five-Seed Head Ablation

Paper role:

- supports the matched-budget comparison among linear, MLP, and KAN heads
- supports the cautious conclusion that KAN is competitive and usually stronger than MLP, but not uniformly better than linear

Public evidence:

- [evidence/ablation/five_seed_head_ablation_summary.md](evidence/ablation/five_seed_head_ablation_summary.md)

Regeneration script:

- [code/wpheadmixer/scripts/run_repeated_ablation.py](code/wpheadmixer/scripts/run_repeated_ablation.py)

## 5. Wavelet Sensitivity

Paper role:

- supports the decomposition on/off comparison
- supports the decomposition-level sensitivity discussion
- supports the wavelet-basis sensitivity discussion

Public evidence:

- [evidence/wavelet_sensitivity/wavelet_sensitivity_summary.md](evidence/wavelet_sensitivity/wavelet_sensitivity_summary.md)

Primary regeneration inputs:

- no-decomposition experiment outputs in the original experiment workspace
- decomposition-level comparison logs in the original experiment workspace
- wavelet-basis sensitivity outputs in the original experiment workspace

## 6. Interpretability / Diagnostic Transparency

Paper role:

- supports branch-level leave-one-out analysis
- supports basis-activation heatmaps
- supports the claim of diagnostic transparency rather than strong physical interpretability

Public evidence:

- [evidence/interpretability/etth2_interpretability_summary.md](evidence/interpretability/etth2_interpretability_summary.md)
- [evidence/interpretability/ettm1_interpretability_summary.md](evidence/interpretability/ettm1_interpretability_summary.md)

Regeneration script:

- [code/wpheadmixer/scripts/interpretability_analysis.py](code/wpheadmixer/scripts/interpretability_analysis.py)

## 7. Manuscript Availability

This public repository does not retain the manuscript source.

Use this when checking:

- which repository files still support the paper's claims after the manuscript source has been removed
- whether the wording of public evidence remains aligned with the released code and scripts

## 8. What Is Not Retained Publicly

Not included in this release:

- manuscript source files
- raw datasets
- checkpoints
- raw training logs
- large intermediate artifacts
- private workspace-only experiment directories

The public repository retains executable scripts plus condensed evidence summaries. A reviewer can reproduce the retained summaries after preparing the datasets locally.
