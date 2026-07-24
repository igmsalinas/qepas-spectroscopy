# QEPAS spectroscopy — CO2 isotope concentration prediction

Machine-learning pipeline to predict **13CO2** and **12CO2** concentrations from QEPAS resampled data.

## Project structure

```
qepas-spectroscopy/
├── data/
│   └── calibration-measures/          # raw measurement folders + Info-Experimento.xlsx
├── docs/                              # experiment documentation (PDF)
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA notebook
│   ├── 02_modeling.ipynb              # traditional modeling notebook
│   └── 03_full_pipeline.ipynb         # full pipeline incl. deep learning
├── qepas_spectroscopy/                # main Python package
│   ├── config.py                      # paths and labels
│   ├── data.py                        # data loading
│   ├── evaluation.py                  # metrics
│   ├── plotting.py                    # figures
│   ├── cli.py                         # command-line interface
│   ├── features/                      # feature engineering
│   └── models/                        # Ridge, RF, XGBoost, Deep CNN
├── outputs/
│   ├── eda/                           # EDA figures
│   ├── features/                      # feature_table.csv + scaler
│   ├── keras_tuner/                   # tuning artifacts
│   └── models/                        # metrics, predictions, parity plots
├── pyproject.toml
└── README.md
```

## Setup

```bash
uv sync
```

## Run

```bash
uv run qepas-train --deep-tuner-trials 15 --deep-epochs 120
```

Options:
- `--skip-deep` — skip Keras CNN
- `--skip-xgb-tune` — skip XGBoost grid search
- `--resume` — resume previous KerasTuner search and reuse already-trained CV fold checkpoints
- `--deep-tuner-trials N` — number of KerasTuner trials
- `--deep-epochs N` — max epochs per CV fold
- `--signal-length N` — downsampled signal length (e.g. 4096, 8192, 16384)
- `--tensorboard-dir DIR` — TensorBoard log directory (defaults to `outputs/tensorboard` and also stores fold checkpoints)

## Data

- 7 concentration levels × 20 scans = 140 samples.
- Per-scan arrays:
  - `modulo_remuestreado.npy` — resampled modulus signal (200,000 samples)
  - `fase_remuestreada.npy` — resampled phase signal
  - `X_remuestreada.npy`, `Y_remuestreada.npy` — quadrature components
  - `vector_med_presion.npy`, `vector_med_flujo.npy`, `vector_temp_Vflujo.npy` — measured pressure, flow, temperature
  - `vector_cons_presion.npy`, `vector_cons_flujo.npy` — set-point pressure/flow
- Concentration labels come from `Info-Experimento.xlsx`.

## Features

Engineered-feature approach (92 features per scan):
- Statistical moments and percentiles of `modulo`, `phase`, `X`, `Y`
- Phase-bin averages of the modulus
- Welch PSD peak frequencies and powers
- Environmental scalars (pressure, flow, temperature)

Deep-learning approach:
- Downsampled raw 4-channel signals (`modulo`, `phase`, `X`, `Y`) to 4096 points
- Normalized per-channel
- Environmental scalars concatenated after CNN

## Models & validation

Models are evaluated with **leave-one-concentration-out cross-validation**.

| Model             | 13CO2 RMSE | 13CO2 R² | 12CO2 RMSE | 12CO2 R² |
|-------------------|------------|----------|------------|----------|
| Ridge Regression  | 0.227      | 0.924    | 20.34      | 0.924    |
| Random Forest     | 0.335      | 0.835    | 29.92      | 0.835    |
| XGBoost           | 0.352      | 0.818    | 30.46      | 0.829    |
| Gradient Boosting | 0.354      | 0.816    | 31.78      | 0.814    |
| Deep CNN          | 0.293      | 0.874    | 39.20      | 0.717    |

**Best model:** Ridge Regression on standardized engineered features.

XGBoost best params: `max_depth=3, learning_rate=0.03, n_estimators=400, subsample=0.8`.

Top engineered features: `phase_p10`, `mod_psd_total`, `mod_peak0_pow`, `mod_psd_max`, `mod_skew`, phase-bin means.

## Outputs

Generated artifacts:
- `outputs/eda/scan_summary.csv`
- `outputs/features/feature_table.csv`
- `outputs/features/raw_dataset.npz`
- `outputs/models/metrics_summary.csv`
- `outputs/models/metrics.json`
- `outputs/models/*_predictions.csv`
- `outputs/models/*_parity.png`
- `outputs/models/*_importance.csv`
- `outputs/models/xgb_tuning.json`
- `outputs/models/deep_tuner_best.json`
