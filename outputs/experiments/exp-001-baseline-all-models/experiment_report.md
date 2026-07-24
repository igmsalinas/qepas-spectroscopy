# Experiment exp-001-baseline-all-models

The permanent report for this run is
`docs/experiments/exp-001-baseline-all-models.md`.

## Outcome

Ridge was the strongest baseline, with R² 0.9199 for both targets. The dominant
failure is regression toward the mean on the lowest and highest held-out
concentration groups. The next planned run is a nested grouped comparison of
tuned Ridge, Partial Least Squares, and a compact spectroscopy feature subset.

See `run_manifest.json`, `splits.csv`, `models/metrics_summary.csv`,
`models/predictions.csv`, `models/group_metrics.csv`, and
`models/ridge_coefficient_stability.csv` for reproducible details.
