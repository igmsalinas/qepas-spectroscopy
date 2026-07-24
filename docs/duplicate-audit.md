# Duplicate-file audit

Audit scope: the entire repository excluding `.git`, `.venv`, and generated
`__pycache__` directories. Files were grouped by size and SHA-256 content.

## Source cleanup

The reorganization initially retained 11 legacy compatibility modules, including
a complete `models/dl` mirror. They were removed, and imports, tests, `main.py`,
and the installed console entry point now reference canonical packages directly.

Current source result:

- zero exact-content duplicate groups across source and tests;
- zero compatibility-facade modules;
- no `models/dl` mirror directory;
- one canonical module tree for each responsibility.

## Metadata cleanup and current exact duplicates

The 1,357 tracked `:Zone.Identifier` Windows metadata files were removed. A
repository-wide verification now finds zero remaining files with that suffix,
and `.gitignore` prevents them from being reintroduced.

After exp-004, the final repository-wide audit inspected 2,061 files and found
18 exact-content groups, 861 redundant instances, and 10.60 MiB (11.11 MB) of
theoretically reclaimable space:

- Eight groups are repeated 136-byte NumPy scalar arrays containing pressure,
  flow, temperature, and set-point values repeated by the acquisition format.
- Six groups are immutable experiment inputs or preprocessing artifacts repeated
  across isolated runs. The largest is four copies of the 3.15 MB deep raw
  dataset. Keeping run inputs self-contained is intentional; interrupted runs
  are retained as an audit trail.
- Four groups are generated tuner metadata: exp-004 fold-local oracle state,
  legacy fold-hyperparameter JSON, trial build configuration, and empty tuner
  state. Their duplication is an expected consequence of isolated searches.

No source/test file and no original large signal array is duplicated.

## Similar-looking artifact folders

- `outputs/keras_tuner` (3.8 MB) is an older tuning experiment.
- `outputs/keras_tuner_advanced` (30 MB) is a separate, newer experiment.
- `outputs/tensorboard/fold_checkpoints` (2.3 MB) contains legacy v1 fold
  checkpoints that the current v4 loader intentionally ignores.
- `outputs/experiments/exp-003-literature-deep-architectures` and
  `exp-003b-literature-deep-conditional` are interrupted diagnostic runs. They
  preserve the discovered tuner-protocol failures and are not result-bearing
  experiments.

These directories are not accidental source mirrors. They were preserved because
deleting experiment history is a separate retention decision. A future artifact
store should deduplicate immutable inputs by content hash while keeping each run
manifest self-contained.
