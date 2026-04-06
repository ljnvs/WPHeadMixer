# WPHeadMixer: Code and Evidence Release

[中文说明 / Chinese Version](README.zh-CN.md)

This repository is the public release package for the WPHeadMixer project:

**WPHeadMixer: A Wavelet-Patch Forecasting Framework with Modular Branch Heads for Long-Term Time Series Forecasting**

It is organized for public release and reproducibility support. The package preserves the materials needed to:

- inspect the released implementation,
- rerun the retained publication-facing scripts after local data preparation,
- trace major paper claims to retained evidence files and regeneration scripts.

This repository is a **curated release snapshot**, not a full private research workspace mirror.

## Repository Layout

- [code/wpheadmixer/](code/wpheadmixer/) - released implementation and publication-facing utility scripts
- [code/wpheadmixer/data/](code/wpheadmixer/data/) - bundled benchmark datasets used by the released paper package
- [evidence/](evidence/) - retained summary evidence files used to support key paper claims
- [paper_evidence_map.md](paper_evidence_map.md) - mapping from major paper claims/tables to repository evidence
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) - minimum publication checklist for this release package

## Environment Setup

Use **Python 3.10**.

Install the core dependencies from the repository root:

```bash
pip install -r code/wpheadmixer/requirements.txt
```

Optional:

- `thop` is only needed if you want to call the FLOPs utility in the training code.

The main experiment entry point is:

- [code/wpheadmixer/run_LTF.py](code/wpheadmixer/run_LTF.py)

## Data Preparation

The six benchmark datasets used by the released paper package are already bundled in this repository:

- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `Weather`
- `Exchange`

They are stored under [code/wpheadmixer/data/](code/wpheadmixer/data/).

For the expected dataset layout, bundled file names, dataset-source notes, and publication-facing reproduction scripts, see:

- [code/wpheadmixer/readme.md](code/wpheadmixer/readme.md)
- [code/wpheadmixer/data/README.md](code/wpheadmixer/data/README.md)

## Dataset Sources

The bundled six-dataset subset follows the standard long-term time series forecasting benchmark setup used in the released code:

- `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`: Electricity Transformer Temperature (ETT) benchmark distributed through the ETT dataset repository and widely used in Informer/Autoformer-style LTSF evaluation.
- `Weather`: weather benchmark CSV distributed in the common LTSF benchmark bundles; the variables correspond to the Jena climate weather measurements commonly used by this benchmark line.
- `Exchange`: exchange-rate benchmark CSV distributed in the common multivariate time-series benchmark collections used by prior LTSF baselines.

The repository bundles the exact CSV files used by the released code package for convenience and reproducibility.

## Reproduction Scope

This release is intended to support:

- inspection of the released code,
- regeneration of the retained summary evidence files,
- verification of how the repository supports the paper's main claims.

This release does **not** include:

- the manuscript source,
- full experiment dumps,
- complete log collections,
- model checkpoints,
- large intermediate outputs,
- local cache files,
- unrelated baseline source trees from the original workspace.

## Included Evidence

The retained evidence files are intentionally compact:

- [evidence/main_results/linear_main_results_summary.md](evidence/main_results/linear_main_results_summary.md)
- [evidence/multi_seed/small_multi_seed_verification_summary.md](evidence/multi_seed/small_multi_seed_verification_summary.md)
- [evidence/ablation/five_seed_head_ablation_summary.md](evidence/ablation/five_seed_head_ablation_summary.md)
- [evidence/wavelet_sensitivity/wavelet_sensitivity_summary.md](evidence/wavelet_sensitivity/wavelet_sensitivity_summary.md)
- [evidence/interpretability/etth2_interpretability_summary.md](evidence/interpretability/etth2_interpretability_summary.md)
- [evidence/interpretability/ettm1_interpretability_summary.md](evidence/interpretability/ettm1_interpretability_summary.md)

For a structured claim-to-evidence index, see:

- [paper_evidence_map.md](paper_evidence_map.md)

## Release Notes

- This release package was cleaned for publication from a larger local research workspace.
- Retained evidence files were sanitized to remove local machine paths from the published copies.
- The release package keeps only the materials needed to inspect the method, code, and supporting evidence.

## License and Third-Party Notice

- The project code retains the license file at [code/wpheadmixer/LICENSE](code/wpheadmixer/LICENSE).
- Vendored wavelet dependency code is retained under [code/wpheadmixer/pytorch_wavelets/](code/wpheadmixer/pytorch_wavelets/), including its own license file.
