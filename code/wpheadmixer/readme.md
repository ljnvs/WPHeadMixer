# WPHeadMixer Code Package

This directory contains the released implementation used by the companion paper repository.

It is **not** a full mirror of the upstream WPMixer project. The contents here were reduced to match the scope of the paper companion release.

## What Is Included

- core training and evaluation entry point: [run_LTF.py](run_LTF.py)
- model code for linear, MLP, KAN, and hybrid branch heads
- utility scripts used to regenerate retained summary evidence
- vendored wavelet dependency code required by the released implementation

## Environment

Use **Python 3.10** and install:

```bash
pip install -r requirements.txt
```

Optional:

- install `thop` only if you want to use the FLOPs helper in `exp/exp_main.py`

## Expected Data Layout

Create a `data/` directory under this folder and place the public benchmark datasets as follows:

```text
data/
  ETT/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
  weather/
    weather.csv
  exchange_rate/
    exchange_rate.csv
  electricity/
    electricity.csv
  traffic/
    traffic.csv
```

The released paper package mainly uses:

- `ETTh1`
- `ETTh2`
- `ETTm1`
- `ETTm2`
- `Weather`
- `Exchange`

## Main Entry Point

The main command-line experiment entry is:

```bash
python run_LTF.py --model WPMixer --head_type linear --data ETTh1 --pred_len 96
```

Key options:

- `--model {WPMixer,WPMixerKAN}`
- `--head_type {linear,mlp,kan,hybrid}`
- `--match_head_params`
- `--head_param_budget`
- `--kan_grid_size`
- `--kan_spline_order`
- `--no_decomposition`

## Publication-Facing Utility Scripts

The retained scripts under [scripts/](scripts/) are the paper-facing reproduction helpers:

- [scripts/run_linear_main_body.py](scripts/run_linear_main_body.py)
  Generates the retained summary for the linear main-body experiment setting.

- [scripts/run_small_seed_verification.py](scripts/run_small_seed_verification.py)
  Regenerates the compact three-seed verification summary retained in the public evidence package.

- [scripts/run_repeated_ablation.py](scripts/run_repeated_ablation.py)
  Runs repeated-seed head ablations and writes machine-readable CSV/Markdown summaries.

- [scripts/interpretability_analysis.py](scripts/interpretability_analysis.py)
  Runs branch-level and KAN-level post-hoc diagnostics for a trained checkpoint.

## Example Commands

Linear main-body summary:

```bash
python scripts/run_linear_main_body.py
```

Compact three-seed verification:

```bash
python scripts/run_small_seed_verification.py
```

Five-seed matched-budget head ablation example:

```bash
python scripts/run_repeated_ablation.py --data ETTh1 --pred_len 96 --heads linear mlp kan --seeds 42 43 44 45 46 --match_head_params
```

Interpretability analysis example:

```bash
python scripts/interpretability_analysis.py --setting_dir ./logs/WPMixer_ETTh2_head-kan_dec-True_sl512_pl96_dm256_bt128_wvdb2_tf5_df5_ptl16_stl8_sd43
```

## Scope Notes

- This release package intentionally excludes checkpoints, full logs, and large experiment outputs.
- Some evidence files included at the repository root were generated from larger local runs and then reduced for publication.
- For the paper-to-evidence mapping, see the repository root file [../paper_evidence_map.md](../paper_evidence_map.md).
