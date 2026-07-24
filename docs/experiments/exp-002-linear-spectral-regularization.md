# Experiment exp-002-linear-spectral-regularization

## Status and hypothesis

Status: completed. This experiment tested whether a low-variance chemometric
calibration model, selected without group leakage, could improve the baseline
Ridge result—especially at the concentration endpoints.

Command:

```bash
uv run qepas-train \
  --run-id exp-002-linear-spectral-regularization \
  --traditional-suite spectral \
  --skip-deep \
  --skip-xgb-tune
```

The 140 selected scans and the same seven concentration/time groups from exp-001
were reused. Each model used seven outer Leave-One-Group-Out folds. Ridge alpha
or PLS components were selected using inner Leave-One-Group-Out validation on
only the current outer fit partition. The outer test labels were never used for
feature scaling, parameter selection, or fitting.

## Results

| Model | Feature view | Target | RMSE | MAE | R² |
|---|---|---:|---:|---:|---:|
| RidgeNestedCompact | 25 features | 13CO2 | 0.1483 | 0.1042 | 0.9677 |
| RidgeNestedCompact | 25 features | 12CO2 | 13.2606 | 9.3137 | 0.9677 |
| PLSNestedCompact | 25 features | 13CO2 | 0.1518 | 0.1100 | 0.9661 |
| PLSNestedCompact | 25 features | 12CO2 | 13.5766 | 9.8358 | 0.9661 |
| RidgeNestedFull | 91 features | 13CO2 | 0.1779 | 0.1276 | 0.9535 |
| RidgeNestedFull | 91 features | 12CO2 | 15.9071 | 11.4102 | 0.9535 |
| PLSNestedFull | 91 features | 13CO2 | 0.1834 | 0.1296 | 0.9506 |
| PLSNestedFull | 91 features | 12CO2 | 16.3954 | 11.5849 | 0.9506 |

Against exp-001 Ridge, compact nested Ridge reduced RMSE by 36.45% and MAE by
42.73% for both targets, raising R² from 0.9199 to 0.9677. Endpoint MAE fell by
about 39%, while interior-group MAE fell by about 46%; the improvement was not
purchased by sacrificing the middle concentrations.

Compact PLS was statistically close: R² 0.9661. It reduced endpoint MAE by about
43% and interior MAE by about 37%. Its residual/actual correlation magnitude
improved from 0.445 to 0.367, whereas compact Ridge's correlation magnitude
increased to 0.473 despite its lower absolute error.

## Selected regularization

- Full Ridge selected alpha 100 in every outer fold.
- Compact Ridge selected alpha 10 in five folds and 100 in two folds.
- Full PLS selected two or three components.
- Compact PLS selected five components in the first three folds, then two, two,
  two, and three components.

The consistent preference for strong Ridge regularization and very few PLS
components confirms that the 91-feature view has more variance than this
seven-condition campaign can support. The 25-feature spectroscopy subset is the
important ablation, not a cosmetic feature reduction.

## Decision

Exp-002 met the error, endpoint, and interior acceptance criteria from exp-001.
`RidgeNestedCompact` is the current accuracy leader. `PLSNestedCompact` is the
stronger secondary model when residual trend matters. Neither is a production
claim because all groups come from one sequential campaign and the two targets
are almost perfectly collinear.

The next experiment was exp-003c: test literature-guided Inception and dilated
residual networks with a much wider conditional hyperparameter space. Its
purpose was to determine whether learned raw-spectral filters add signal beyond
the compact calibration model—not to assume that more depth would help.
