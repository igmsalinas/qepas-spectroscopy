# Engineering review

## Resolved in this refactor

### Evaluation correctness

- Ridge standardization was fitted on the full dataset before cross-validation. It now lives inside the cross-validated sklearn pipeline.
- Deep signal, scalar, and target normalizers were fitted on all samples. They are now fitted per fold on fit groups only.
- The outer DL test group was used for early stopping. Each outer fold now has a separate rotating inner validation group.
- Fold resume restored weights without the preprocessing state that produced them. Versioned checkpoints now persist fold identity, hyperparameters, and all normalizers.
- Default-CNN checkpoint saving dereferenced a missing tuner and could fail. Default and tuned model specifications now have separate persistence paths.
- KerasTuner resume checked only `trial_0`, while the project uses padded names such as `trial_00`. Resume now detects any `trial_*` directory.
- Learning-rate schedules interpreted epochs as optimizer steps. They now use the actual steps per epoch.
- The Transformer tuner branch passed an unsupported argument and could not build. Every registered architecture now has a tested common input/output contract.
- TCN and LSTM normalization choices were sampled but ignored. Their builders now apply the selected policy.
- XGBoost tuning used only the first held-out concentration. It now scores every leave-one-group-out split.

### Design and testability

- The package is separated into application, core, data, validation, evaluation, visualization, model-family, and interface layers.
- CLI orchestration moved into an injectable, framework-independent `TrainingPipeline`.
- Dataset assembly moved out of the CLI into an injectable, single-pass `DatasetPipeline`.
- Validated dataset value objects enforce dimensionality, alignment, and finite numeric inputs.
- Group-splitting policy is isolated in a pure module and covered by disjointness tests.
- Scan I/O reports all missing required arrays at the boundary instead of failing later with a `KeyError`.
- Raw signals use exact-length anti-aliased resampling; phase is circularly encoded instead of interpolated across wraps.
- Traditional model imports no longer eagerly initialize TensorFlow.
- Evaluation rejects invalid shapes and non-finite predictions before calling metric libraries.
- The fully nested small-data deep protocol now tunes inside every outer fold,
  evaluates paired seeds, applies augmentation only to fit partitions, and
  persists signature-validated partial fold predictions for safe resume.
- Stochastic ensembles persist aggregate, per-seed, per-group, and
  seed-by-group diagnostics rather than hiding initialization variance.

## Remaining risks and recommended next work

1. **Independent scientific validation.** All samples come from one calibration campaign. Add groups for acquisition day, instrument, operator, and gas preparation, then hold out an entire campaign for final acceptance.
2. **Protocol selection and external validity.** The default `development` deep protocol still reuses one tuning result and must not be reported as a fully independent model-selection estimate. Use `nested-small-data` for internal selection estimates. Even fully nested folds cannot replace a final untouched campaign.
3. **Deployable model artifact.** The pipeline evaluates cross-validation folds but does not train and package one final production model with input schema, feature version, label provenance, and prediction API. Add a separate `fit-final` workflow after model selection.
4. **Label provenance.** Runtime labels are hardcoded. Parse and validate the experiment workbook (or a versioned manifest), record units, and fail when folder/time mappings are ambiguous.
5. **Target semantics.** In this dataset 12CO2 and 13CO2 are deterministically related by the mixture preparation. Confirm whether two independent outputs are scientifically meaningful; a constrained or single-latent-target model may be more appropriate.
6. **Repository size and provenance.** The initial commit tracks hundreds of generated artifacts, large NumPy files, while the 1,357 `Zone.Identifier` files have now been removed. Move raw/derived data and model artifacts to DVC, object storage, or an experiment registry; keep checksums and small manifests in Git.
7. **Experiment tracking.** MLflow is installed but unused. Record code revision, data manifest, split assignments, hyperparameters, normalizers, metrics, and artifacts for every run.
8. **Broader tests.** Add a small end-to-end CLI fixture, checkpoint corruption tests, trained-model smoke tests for every architecture, and golden tests for feature extraction before changing scientific transforms.
