# Calibration data analysis

The reusable `DatasetProfiler` streamed over all 150 discovered scans. The raw
arrays were read-only memory maps; no measurement was changed. Re-run the exact
analysis with:

```bash
uv run qepas-profile
```

The machine-readable results are in
[`outputs/eda/data_profile.json`](../outputs/eda/data_profile.json), with one row
per acquisition in
[`outputs/eda/scan_profile.csv`](../outputs/eda/scan_profile.csv).

## Findings

- Seven concentration groups were found. Six contain 20 acquisitions; the
  `13:43:08` group contains 30. Training uses a deliberate 20-per-group cap,
  yielding a balanced 140-sample cohort, while profiling covers all 150.
- All four signal arrays have exactly 200,000 `float64` values in every scan.
  No NaN or infinity was found.
- Stored modulus agrees with `hypot(X, Y)` at mean relative MAE
  `1.337e-6`. Stored phase agrees with `atan2(X, Y)` at circular MAE
  `0.00280 rad`. X/Y are therefore redundant with modulus/phase for this
  campaign.
- Phase has between 11 and 171 wrap jumps per scan (median 42.5). Direct phase
  interpolation and ordinary linear statistics are discontinuity-sensitive.
- Per-scan modulus mean, standard deviation, and maximum correlate with 13CO2 at
  `r=0.984`, `0.979`, and `0.984`. This is why the pipeline preserves absolute
  amplitude and rejects per-sample z-scoring.
- `P_cons` and `F_cons` are constant. Measured pressure has three unique values,
  measured flow three, and temperature two. The normalizer safely maps
  zero-variance features to zero, but these controls add no discrimination in
  the current campaign.
- 13CO2 and 12CO2 labels have correlation effectively equal to one. This dataset
  cannot demonstrate independent isotope-ratio prediction; it principally tests
  total mixture concentration. A future campaign must vary isotope ratio
  independently if two-output identifiability is a requirement.
- Each time/group has exactly one label pair, and the seven pairs rise
  monotonically from `[0, 0]` to `[2.4742, 221.2258]`. Acquisition time, group,
  and concentration are perfectly confounded. Grouped validation prevents
  replicate leakage but cannot distinguish time drift from concentration.

## Pipeline decisions

1. Validate required files, numeric dtypes, finite values, scalar cardinality,
   equal signal lengths, sample identifiers, and cross-view ordering.
2. Build engineered and deep-learning views in one scan pass.
3. Wrap phase for engineered bins and add circular phase statistics.
4. For deep models, filter and resample modulus plus sine/cosine phase channels.
   Polyphase resampling is the default to suppress aliasing.
5. Retain X/Y only as an explicit compatibility/ablation option.
6. Fit signal, scalar, target, and Ridge feature normalization only on training
   partitions. Persist normalization state and a preprocessing fingerprint with
   model checkpoints.
7. Split by concentration group; never split replicate scans randomly.

## Scientific limitations

The correlations above are descriptive, not performance estimates: concentration
and acquisition time are confounded, all data comes from one day, and there are
only seven distinct label pairs. Acceptance testing needs independent days,
instruments, operators, gas preparations, and isotope ratios.
