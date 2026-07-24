# QEPAS data profile

Generated from the calibration arrays without modifying them.

## Inventory

- Discovered scans: 150
- Concentration groups: 7
- Balanced training selection: 140 scans (20 per group)
- Signal lengths: [200000]
- Signal dtypes: ['float64']
- Non-finite values: 0

| Group | Scans |
|---|---:|
| 12:46:00 | 20 |
| 12:55:42 | 20 |
| 13:05:15 | 20 |
| 13:14:37 | 20 |
| 13:24:02 | 20 |
| 13:33:30 | 20 |
| 13:43:08 | 30 |

## Signal integrity

- Relative MAE between stored modulus and `hypot(X, Y)`: 1.33704349e-06
- Circular MAE between stored phase and `atan2(X, Y)`: 0.00280141689 radians
- Phase-wrap jumps per scan (median): 42.5

## Target structure

The correlation between 13CO2 and 12CO2 labels is
1.000000000000. The campaign therefore varies total mixture
concentration but does not independently vary isotope ratio.

## Correlation with 13CO2

These values are descriptive only; group-disjoint validation remains mandatory.

| Per-scan statistic | Pearson r |
|---|---:|
| mod_mean | 0.983974 |
| mod_std | 0.978621 |
| mod_max | 0.983868 |
| phase_sin_mean | -0.955074 |
| phase_cos_mean | 0.985264 |
| P_med | -0.148853 |
| F_med | 0.118138 |
| Temp | -0.227904 |
