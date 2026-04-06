# Wavelet Sensitivity Summary

This summary retains the wavelet-related evidence supporting the paper's decomposition and basis-sensitivity discussion.

## 1. Decomposition On vs Off

Retained setting:

- Head: `kan`
- Wavelet: `db2`
- Level: `1`
- Horizon: `pred_len=96`

| Dataset | With decomposition MSE / MAE | Without decomposition MSE / MAE | Observation |
| --- | ---: | ---: | --- |
| ETTh1 | 0.366294 / 0.395762 | 0.364418 / 0.394770 | Disabling decomposition is slightly better in this retained run. |
| ETTm1 | 0.301742 / 0.349586 | 0.305676 / 0.356654 | Decomposition improves both MSE and MAE. |
| Weather | 0.148062 / 0.193229 | 0.149588 / 0.194131 | Decomposition improves both MSE and MAE. |

Interpretation:

- The decomposition module is beneficial on ETTm1 and Weather in the retained comparison.
- ETTh1 shows only a very small difference, so the paper should avoid claiming a universally large gain from decomposition.

## 2. Decomposition Level Sensitivity

Retained setting:

- Head: `kan`
- Wavelet: `db2`
- Horizon: `pred_len=96`

| Dataset | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Best retained level |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh1 | 0.366294 / 0.395762 | 0.363502 / 0.393523 | 0.367960 / 0.395904 | 0.377552 / 0.403967 | 0.387408 / 0.407250 | Level 2 |
| ETTm1 | 0.301742 / 0.349586 | 0.301308 / 0.351790 | 0.291304 / 0.348109 | 0.295920 / 0.346422 | 0.309566 / 0.355650 | Level 3 by MSE, Level 4 by MAE |
| Weather | 0.147421 / 0.191820 | 0.146753 / 0.191693 | 0.151178 / 0.197762 | 0.151264 / 0.194919 | 0.157052 / 0.205011 | Level 2 |

Interpretation:

- Very deep decomposition is consistently harmful in the retained runs.
- Moderate levels work best: Level 2 on ETTh1 and Weather, Level 3 or 4 on ETTm1 depending on metric.

## 3. Wavelet Basis Sensitivity

Retained setting:

- Head: `kan`
- Level: `1`
- Horizon: `pred_len=96`

### ETTh1

| Basis | MSE / MAE |
| --- | ---: |
| bior2.4 | 0.364521 / 0.394365 |
| coif1 | 0.370312 / 0.397792 |
| coif2 | 0.362877 / 0.393827 |
| db1 | 0.365352 / 0.393909 |
| db2 | 0.366294 / 0.395762 |
| db4 | 0.364658 / 0.393135 |
| db8 | 0.363743 / 0.394415 |
| rbio2.4 | 0.365530 / 0.395816 |
| sym2 | 0.366294 / 0.395762 |
| sym4 | 0.365326 / 0.394727 |

Best retained ETTh1 basis:

- `coif2` by MSE
- `db4` by MAE

### ETTm1

| Basis | MSE / MAE |
| --- | ---: |
| bior2.4 | 0.298697 / 0.348439 |
| coif1 | 0.303699 / 0.349574 |
| coif2 | 0.302133 / 0.355450 |
| db1 | 0.309618 / 0.352091 |
| db2 | 0.301742 / 0.349586 |
| db4 | 0.301438 / 0.350429 |
| db8 | 0.294437 / 0.348053 |
| rbio2.4 | 0.299229 / 0.347714 |
| sym2 | 0.301742 / 0.349586 |
| sym4 | 0.305815 / 0.351401 |

Best retained ETTm1 basis:

- `db8` by MSE
- `rbio2.4` by MAE

### Weather

| Basis | MSE / MAE |
| --- | ---: |
| bior2.4 | 0.151795 / 0.197198 |
| coif1 | 0.146951 / 0.191835 |
| coif2 | 0.147954 / 0.192277 |
| db1 | 0.148487 / 0.195168 |
| db2 | 0.148062 / 0.193229 |
| db4 | 0.148737 / 0.193346 |
| db8 | 0.147583 / 0.192144 |
| rbio2.4 | 0.151009 / 0.196190 |
| sym2 | 0.148126 / 0.193285 |
| sym4 | 0.149248 / 0.193765 |

Best retained Weather basis:

- `coif1` by both MSE and MAE

## Overall conclusions

- Performance varies with wavelet basis and decomposition depth, so the paper's wording should remain dataset-specific.
- There is no single basis that dominates all retained datasets.
- The retained evidence supports a robustness claim, not a claim that one fixed wavelet choice is universally optimal.
