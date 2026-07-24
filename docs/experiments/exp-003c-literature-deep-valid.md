# Experiment exp-003c-literature-deep-valid

## Status and hypothesis

Status: completed. The experiment asked whether literature-guided multi-scale or
dilated convolution can learn useful raw QEPAS spectral filters beyond the
simple CNN baseline.

Two diagnostic runs preceded it and remain in the experiment registry:

- `exp-003-literature-deep-architectures` exposed architecture parameters that
  were incorrectly active for unrelated trials. Conditional KerasTuner scopes
  were added and the run was marked interrupted.
- `exp-003b-literature-deep-conditional` exposed an invalid comparison of
  `val_loss` across MSE, MAE, Huber, and LogCosh trials. Ranking and early
  stopping were changed to normalized-target `val_mae`; the run was marked
  interrupted.

The valid command was:

```bash
uv run qepas-train \
  --run-id exp-003c-literature-deep-valid \
  --skip-traditional \
  --skip-xgb-tune \
  --architectures inception_spectra \
  --architectures dilated_resnet1d \
  --deep-tuner-trials 12 \
  --deep-tuner-epochs 30 \
  --deep-tuner-executions 1 \
  --deep-epochs 60 \
  --deep-batch-size 8 \
  --deep-early-stopping-patience 10 \
  --signal-length 2048
```

## Search result

The best single development-group trial reached normalized validation MAE
0.1862. It selected six Inception modules, 32 filters, 32 bottleneck filters,
maximum kernel 63, stem stride 2, batch normalization, GELU, dropout 0.2,
AdamW at learning rate 0.00303 with exponential decay, weight decay 0.00128,
and LogCosh loss.

Ten Inception and two dilated residual trials completed. The best Inception and
dilated scores were 0.1862 and 0.3718; their median scores were 0.5456 and
0.4133. This uneven Bayesian allocation is sufficient to select a development
candidate, not to claim that one architecture family is universally superior.

## Grouped evaluation

| Model | Target | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Selected Inception | 13CO2 | 0.5012 | 0.4188 | 0.6307 |
| Selected Inception | 12CO2 | 49.9092 | 39.7688 | 0.5419 |

Relative to exp-001 SimpleCNN, RMSE worsened by 62.37% for 13CO2 and 77.24% for
12CO2. Relative to exp-002 compact nested Ridge, RMSE worsened by 237.93% and
276.37%. The worst held group was `13:33:30`: RMSE was 0.8116 for 13CO2 and
88.4906 for 12CO2, with strong positive bias. The model also had large positive
bias on `12:46:00` and `13:14:37`. This is unstable group-to-group calibration,
not a small aggregate-metric miss.

The selected hyperparameters were tuned once using `13:43:08` as the development
group and reused across all seven outer fits. Each outer fit still owns its
normalizers and a different group-disjoint early-stopping set, but architecture
selection is not nested per outer fold. The fold testing `13:43:08` is therefore
not an independent hyperparameter test. Since the aggregate result is already
far below the baselines, this possible optimism does not alter the rejection.

## Dataset split answer

There is no permanent random train/validation/test file split. The run records:

- one six-group/one-group development split for KerasTuner;
- seven evaluation folds, each with five fit groups (100 scans), one validation
  group (20 scans), and one test group (20 scans);
- one out-of-fold test prediction per selected scan.

All data still comes from one sequential measurement campaign. There is no
untouched day- or campaign-level test set.

## Decision and next experiment

Reject this Inception configuration for model promotion. More depth and a wider
search did not solve the governing problem: only seven independent conditions
are available, and the tuning group is not representative of all held groups.
The current champion remains `RidgeNestedCompact` from exp-002.

Exp-004 should test a data-efficiency hypothesis rather than another capacity
increase:

1. Preferred: acquire a second campaign with randomized concentration order,
   independently varied isotope ratios, and the first campaign frozen for
   development. Hold the second campaign untouched for final testing.
2. If acquisition is temporarily impossible, constrain Inception to two or
   three modules with 8–16 filters and tune separately inside every outer fold.
3. Evaluate five random seeds (the InceptionTime paper uses a five-model
   ensemble) and report the seed distribution before considering an ensemble.
4. Add one fit-only augmentation ablation using measured-noise injection plus
   small physically justified gain/baseline perturbations. Do not shift spectra
   until wavelength/frequency drift has been quantified.
5. Compare against compact Ridge/PLS on exactly the same outer groups. Require
   at least R² 0.90, improvement over exp-001 SimpleCNN, no held group with
   normalized RMSE above 0.20, and acceptable seed variance before continuing.
6. Prefer simulation pretraining or a cross-fitted CNN-feature/PLS hybrid over
   Transformers or a deeper CNN if a validated forward model becomes available.

Without a new campaign, exp-004 remains a development study; it cannot remove
the central external-validity limitation.
