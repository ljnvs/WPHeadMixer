# Five-Seed Head Ablation Summary

This summary retains the matched-budget, five-seed head ablation evidence used to support the paper's head-comparison claims.

## Shared setting

- Horizon: `pred_len=96`
- Heads: `linear`, `mlp`, `kan`
- Seeds: `42, 43, 44, 45, 46`
- Report format: `mean +/- std`

## Summary table

| Dataset | Linear MSE | Linear MAE | MLP MSE | MLP MAE | KAN MSE | KAN MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.379674 +/- 0.003010 | 0.406224 +/- 0.003050 | 0.406770 +/- 0.015655 | 0.428976 +/- 0.008778 | 0.401333 +/- 0.013052 | 0.424105 +/- 0.008942 |
| ETTh2 | 0.291116 +/- 0.010746 | 0.350366 +/- 0.006327 | 0.296788 +/- 0.003328 | 0.354542 +/- 0.002647 | 0.295182 +/- 0.010691 | 0.352278 +/- 0.004939 |
| ETTm1 | 0.304293 +/- 0.006622 | 0.353107 +/- 0.007197 | 0.309815 +/- 0.004547 | 0.362731 +/- 0.003791 | 0.303259 +/- 0.004514 | 0.355201 +/- 0.002376 |
| Weather | 0.147724 +/- 0.001057 | 0.192564 +/- 0.001405 | 0.149378 +/- 0.001722 | 0.194062 +/- 0.002084 | 0.150894 +/- 0.001880 | 0.196121 +/- 0.002036 |

## Evidence-backed conclusions

- MLP is the weakest head on all four retained datasets under this matched-budget setting.
- Linear is best on ETTh1, ETTh2, and Weather.
- KAN achieves the best retained MSE on ETTm1, but it is not uniformly better than Linear across datasets.
- The paper should therefore describe KAN as competitive and selectively advantageous, not consistently dominant.

## Regeneration

- Script: `code/wpheadmixer/scripts/run_repeated_ablation.py`
- Public release note: raw logs and checkpoints are not bundled in the release package.
