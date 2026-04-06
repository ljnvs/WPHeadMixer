# Interpretability Summary

- Setting: `WPMixer_ETTm1_head-kan_dec-True_sl512_pl96_dm256_bt128_wvdb2_tf5_df5_ptl16_stl8_sd43`
- Checkpoint: `Removed from GitHub release package; only summary metrics are retained.`
- Analyzed batches: 16
- Full-prediction sample MSE: 0.256952 ± 0.171234

## Branch-Level Results

| Branch | Energy share | Leave-one-out delta MSE |
| --- | ---: | ---: |
| approx | 0.998545 ± 0.000363 | 0.508321 ± 0.337245 |
| detail_1 | 0.001455 ± 0.000363 | 0.000329 ± 0.000464 |

## KAN-Level Results

| Branch | Activation-error correlation |
| --- | ---: |
| approx | -0.000557 |
| detail_1 | 0.031039 |
