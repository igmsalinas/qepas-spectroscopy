# Experiment exp-001-baseline-all-models

## Status and configuration

- Status: completed
- Started: 2026-07-24 13:25:29 UTC
- Completed: 2026-07-24 13:27:44 UTC
- Selected samples: 140
- Concentration/time groups: 7, with 20 selected scans per group
- Traditional models: Ridge, Random Forest, XGBoost, Gradient Boosting
- Deep model: SimpleCNN
- Deep tuner: 3 trials, 20 epochs per trial
- Deep grouped evaluation: up to 40 epochs per outer fold, patience 6
- XGBoost grid search: skipped for this baseline
- Raw-signal representation: 4,096 polyphase samples with modulus,
  phase-sine, and phase-cosine channels plus five environmental scalars

Command:

```bash
uv run qepas-train \
  --run-id exp-001-baseline-all-models \
  --deep-tuner-trials 3 \
  --deep-tuner-epochs 20 \
  --deep-epochs 40 \
  --deep-early-stopping-patience 6 \
  --architectures simple_cnn \
  --skip-xgb-tune
```

## Split policy

The dataset is not divided once into fixed train, validation, and test files.
Its seven concentration levels were measured sequentially, so the timestamp is
also the concentration-group identifier.

- Traditional outer evaluation: seven Leave-One-Group-Out folds. Each fold has
  120 fit samples and 20 test samples.
- Deep tuning holdout: 120 fit samples and 20 validation samples.
- Deep outer evaluation: seven nested grouped folds. Each fold has 100 fit,
  20 validation, and 20 test samples.

Every selected sample is tested exactly once in each model family's outer
evaluation, and all reported aggregate metrics use these out-of-fold
predictions. There is no separate untouched measurement campaign. The current
metrics estimate interpolation/extrapolation across concentration groups within
one campaign; they do not establish generalization to a new day, instrument,
or environmental regime.

The SimpleCNN hyperparameters were selected once on the `13:43:08` development
group and then reused across the seven deep folds. Preprocessing and early
stopping are group-disjoint, but architecture selection is not nested inside
each outer fold. Deep metrics are therefore development estimates, not an
unbiased final model-selection estimate.

The run-local `splits.csv` records every sample assignment.

## Results

| Model | Target | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Ridge | 13CO2 | 0.2334 | 0.1819 | 0.9199 |
| Ridge | 12CO2 | 20.8661 | 16.2638 | 0.9199 |
| DeepLearning | 13CO2 | 0.3087 | 0.2511 | 0.8599 |
| DeepLearning | 12CO2 | 28.1586 | 22.9230 | 0.8542 |
| RandomForest | 13CO2 | 0.3328 | 0.3117 | 0.8372 |
| RandomForest | 12CO2 | 29.7559 | 27.8705 | 0.8372 |
| XGBoost | 13CO2 | 0.3422 | 0.3157 | 0.8279 |
| XGBoost | 12CO2 | 30.3705 | 27.5143 | 0.8304 |
| GradientBoosting | 13CO2 | 0.3596 | 0.3399 | 0.8099 |
| GradientBoosting | 12CO2 | 31.7637 | 29.3986 | 0.8145 |

Ridge also had the best group-macro normalized RMSE, 0.0889, and the best
worst-group normalized RMSE, 0.1585. DeepLearning ranked second at 0.1115 and
0.2413 respectively.

## Diagnosis

1. The linear baseline is the strongest model. Both targets have almost the
   same relative metrics because the campaign labels are nearly perfectly
   collinear.
2. All models regress toward the mean. Ridge residuals correlate negatively
   with the actual target (-0.445).
3. Endpoint groups dominate error. Ridge endpoint MAE is 0.2767 for 13CO2 and
   24.7435 for 12CO2, versus 0.1440 and 12.8720 on the five interior groups.
4. Tree ensembles cannot extrapolate beyond the target range represented in
   each training fold. Their endpoint biases are therefore expected, not
   primarily a tuning problem.
5. SimpleCNN has the same failure mode and is especially weak on the highest
   group: 12CO2 RMSE is 53.3723 with bias -52.9315. A larger neural search is
   not the highest-value next action with only 140 samples.
6. Phase quantiles/circular summaries and amplitude/PSD features consistently
   rank highly. Ridge fold coefficients are especially stable for
   `mod_median`, `phase_median`, `phase_sin_mean`, `X_std`, `mod_mean`, and
   `Y_mean`.

## Next experiment: exp-002-linear-spectral-regularization

Hypothesis: a lower-variance latent linear calibration model, tuned without
group leakage, will reduce endpoint shrinkage while retaining the interior
accuracy of Ridge.

Planned changes:

1. Keep the exact same seven outer held-out groups for a paired comparison.
2. Add Partial Least Squares regression with component counts
   `{2, 3, 5, 8, 12}`.
3. Tune Ridge alpha over
   `{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`.
4. Compare all 91 engineered features with a compact spectroscopy subset built
   from stable amplitude, phase/circular, PSD, pressure, flow, and temperature
   summaries.
5. Select every hyperparameter only inside each outer training partition using
   inner grouped cross-validation. The outer test group must remain unseen.
6. Report global RMSE/MAE/R², group-macro normalized RMSE, worst-group error,
   bias, endpoint MAE, and the paired per-sample error difference versus this
   baseline.
7. Treat a one-latent-target reconstruction of both collinear labels as an
   explicit ablation, not as an independent two-isotope predictor.

Acceptance criteria:

- improve mean R² beyond 0.9199 on the same nested outer predictions;
- reduce endpoint MAE by at least 15%;
- do not worsen interior-group MAE by more than 5%;
- reduce the magnitude of residual/target correlation;
- retain fold-stable features or latent components.

If this experiment does not meet those criteria, the next priority should be a
new measurement campaign with randomized concentration order and an untouched
campaign-level test set, rather than more architecture search.
