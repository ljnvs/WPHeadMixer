# Interpretability Summary

- Setting: `WPMixer_ETTh2_head-kan_dec-True_sl512_pl96_dm256_bt128_wvdb2_tf5_df5_ptl16_stl8_sd43`
- Checkpoint: `Removed from GitHub release package; only summary metrics are retained.`
- Analyzed batches: 16
- Full-prediction sample MSE: 0.239095 ± 0.219358

## Branch-Level Results

| Branch | Energy share | Leave-one-out delta MSE |
| --- | ---: | ---: |
| approx | 0.979111 ± 0.014617 | 0.039488 ± 0.148900 |
| detail_1 | 0.020889 ± 0.014617 | 0.001747 ± 0.001557 |

## KAN-Level Results

| Branch | Activation-error correlation |
| --- | ---: |
| approx | -0.129628 |
| detail_1 | 0.005010 |
