# QEPAS spectroscopy — CO2 isotope concentration prediction

Machine-learning pipeline that predicts **13CO2** and **12CO2** concentrations from QEPAS resampled signals and environmental measurements.

## Architecture

The implementation is organized into explicit layers:

```text
qepas_spectroscopy/
├── application/              # framework-independent training use cases
├── core/                     # configuration, paths, labels, random seed
├── data/                     # I/O, datasets, normalization, profiling, augmentation
├── features/                 # engineered and raw-signal transformations
├── validation/               # grouped split policies
├── evaluation/               # metrics and result persistence
├── visualization/            # plotting adapters
├── models/
│   ├── traditional/          # factories, grouped trainer, XGBoost tuning
│   └── deep_learning/        # architectures, tuning, schedules, training
└── interfaces/               # Typer CLI adapter
```

`TrainingPipeline` owns orchestration without depending on Typer. `DatasetPipeline`
and `TrainingPaths` are injected dependencies, so application behavior can be
tested using in-memory data and temporary artifact directories.

Source code has one canonical location per responsibility; legacy mirror
modules were removed to prevent drift. See
[`docs/architecture.md`](docs/architecture.md) for dependency rules and examples.

## Setup

```bash
uv sync
```

## Test

Pytest is declared in the development dependency group:

```bash
uv run pytest -q
```

The test modules remain compatible with `unittest` discovery when needed.

## Run

```bash
uv run qepas-profile
uv run qepas-train \
  --experiment-label baseline \
  --deep-tuner-trials 15 \
  --deep-tuner-epochs 40 \
  --deep-epochs 120
```

`qepas-profile` streams across every discovered scan and writes reproducible
CSV, JSON, and Markdown data-quality artifacts without training a model.
Every `qepas-train` invocation creates a new immutable directory under
`outputs/experiments/<experiment-id>/`. Use `--run-id NAME` when an externally
meaningful identifier is required; an existing run is never overwritten.

Useful options:

- `--experiment-label LABEL` — suffix for an automatically generated run ID.
- `--run-id ID` — explicit unique experiment ID.
- `--experiments-dir DIR` — alternative root for isolated experiment runs.
- `--skip-deep` — skip Keras models.
- `--skip-traditional` — skip traditional models and run only deep learning.
- `--traditional-suite baseline|spectral|all` — fixed baselines, nested
  Ridge/PLS calibration, or both.
- `--skip-xgb-tune` — skip the grouped XGBoost grid search.
- `--resume` — resume the explicit `--run-id`; it is rejected for generated IDs.
- `--deep-tuner-trials N` — number of KerasTuner trials.
- `--deep-tuner-epochs N` — maximum epochs per tuner trial.
- `--deep-tuner-executions N` — repeated initializations averaged per trial.
- `--deep-protocol development|nested-small-data` — select once on a
  development group or tune independently in every outer fold.
- `--deep-seeds N` — seed count for the fully nested small-data ensemble.
- `--deep-augmentation-ablation` — pair unaugmented and fit-only augmented
  ensembles with identical fold hyperparameters and initialization seeds.
- `--deep-epochs N` — maximum epochs per deep-learning fold.
- `--architectures NAME` — repeat to restrict the search. Registered choices
  are `simple_cnn`, `resnet1d`, `tcn`, `lstm`, `multiscale_cnn`,
  `transformer1d`, `inception_spectra`, and `dilated_resnet1d`.
- `--signal-length N` — fixed signal length, such as 4096 or 8192.
- `--resampling-method polyphase|linear` — anti-aliased production resampling or legacy linear interpolation.
- `--include-cartesian-signals` — additionally feed redundant X/Y channels.
- `--tensorboard-dir DIR` — TensorBoard and fold-checkpoint directory.

Legacy DL fold checkpoints did not contain preprocessing state and are deliberately ignored. They will be rebuilt on the next resumed run.

## Data and preprocessing

The campaign contains 150 discovered scans across seven concentration levels.
Six groups contain 20 scans and the highest-concentration group contains 30.
Training deliberately selects 20 numerically ordered scans per group (140 total)
so per-sample losses and metrics do not overweight the last concentration.
Profiling still inspects all 150 scans.

Every scan contains four finite 200,000-point `float64` arrays plus measured and
set-point pressure/flow and measured temperature. `DatasetPipeline` validates
this schema and builds both model views in one I/O pass:

- 91 engineered statistics, circular phase summaries, phase-bin summaries,
  Welch-spectrum features, and environmental scalars;
- three compact deep-learning channels: modulus, sine of phase, and cosine of
  phase, plus five environmental scalars.

Phase is encoded as sine/cosine because all scans contain phase wraps. Signals
are reduced with polyphase low-pass resampling to prevent aliasing. X/Y channels
are omitted by default because modulus and phase reproduce them to within the
measured numerical tolerance; they remain available through
`--include-cartesian-signals`.

Amplitude is not normalized per scan: modulus summaries correlate strongly with
concentration and per-sample scaling would erase that information. Deep signal,
scalar, and target standardizers are fitted only on each training fold. Ridge
owns its fold-local `StandardScaler`; tree models receive finite engineered
features directly because scale normalization does not benefit their splits.

See [`docs/data-analysis.md`](docs/data-analysis.md) and the generated
[`outputs/eda/data_profile.md`](outputs/eda/data_profile.md).

## Validation policy

Fixed traditional baselines use leave-one-concentration-out cross-validation.
The spectral suite selects Ridge alpha and PLS components with inner
leave-one-group-out folds inside every outer fold. Its compact/full feature
ablation is therefore a genuinely nested model-selection estimate.

Deep models use nested grouped partitions for evaluation:

1. One concentration group is held out as the untouched outer test fold.
2. A different concentration group is used for early stopping.
3. Normalizers and model weights are fitted only on the remaining groups.

The `development` deep protocol uses one development-group holdout and reuses
the chosen architecture across outer folds; it is explicitly reported as a
development estimate. The `nested-small-data` protocol tunes a constrained
Inception model separately inside every outer fold, supports five-seed
ensembles, and can run a paired fit-only augmentation ablation. In both
protocols, fitted preprocessing and early stopping remain group-disjoint.
Learning-rate schedules convert epochs to optimizer steps using the actual
batch count. Fully nested internal validation still does not replace a separate
campaign-level test set.

This is grouped cross-validation, not one fixed train/validation/test split.
Each sample receives an out-of-fold test prediction, but the current repository
does not contain a separate untouched measurement campaign for final testing.
Each run writes `splits.csv` with the exact sample/fold/role assignments.

Metrics produced before this validation refactor are not comparable and are intentionally not presented here. Regenerate them with the current pipeline before drawing model-quality conclusions.

## Outputs

Each experiment directory contains:

- `run_manifest.json` with status, timestamps, options, data counts, and metrics
- `splits.csv` with the exact grouped fit/validation/test assignments
- `features/feature_table.csv`
- `features/raw_dataset.npz`
- `features/scaler_mean.csv`, `scaler_scale.csv`, and `preprocessing_manifest.json` as inference artifacts
- `models/metrics_summary.csv` and `metrics.json`
- `models/predictions.csv` and `group_metrics.csv`
- `models/parity_grid.png` and `metrics_comparison.png`
- aggregate tree-model feature-importance CSV/PNG files
- `models/xgb_tuning.json`
- `models/nested_traditional_search.json`
- `models/deep_tuner_best.json` for the development protocol
- `models/deep_nested_search.json`, seed predictions, aggregate seed metrics,
  and seed-by-group metrics for the fully nested protocol
- resumable nested-fold prediction blocks and TensorBoard logs

The dataset profiler remains shared under `outputs/eda/` because it describes
source data rather than a model run.

See [`docs/engineering-review.md`](docs/engineering-review.md) for resolved
findings, [`docs/duplicate-audit.md`](docs/duplicate-audit.md) for the
repository-wide content audit, and
[`docs/research/deep-learning-spectroscopy.md`](docs/research/deep-learning-spectroscopy.md)
for the architecture literature review. Completed run decisions and commands
are indexed in [`docs/experiments/README.md`](docs/experiments/README.md).
