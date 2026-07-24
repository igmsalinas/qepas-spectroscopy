# Experiment registry

| ID | Status | Main result | Decision |
|---|---|---|---|
| [exp-001-baseline-all-models](exp-001-baseline-all-models.md) | completed | Ridge R² 0.9199; SimpleCNN R² 0.8599/0.8542 | Establish paired grouped baseline. |
| [exp-002-linear-spectral-regularization](exp-002-linear-spectral-regularization.md) | completed | Compact nested Ridge R² 0.9677; compact PLS R² 0.9661 | Compact Ridge is current accuracy leader. |
| `exp-003-literature-deep-architectures` | interrupted diagnostic | Revealed non-conditional architecture parameters. | Fix tuner space; do not compare metrics. |
| `exp-003b-literature-deep-conditional` | interrupted diagnostic | Revealed incomparable `val_loss` objective across loss families. | Rank trials by normalized `val_mae`; do not compare metrics. |
| [exp-003c-literature-deep-valid](exp-003c-literature-deep-valid.md) | completed | Broad Inception R² 0.6307/0.5419 | Reject configuration; reduce capacity and nest selection. |
| [exp-004-fully-nested-small-inception](exp-004-fully-nested-small-inception.md) | completed | Augmented five-seed Inception R² 0.8969/0.8900 | Augmentation works, but compact Ridge remains leader; acquire an independent campaign. |

The machine-readable comparison is
[`outputs/experiments/experiment_comparison.csv`](../../outputs/experiments/experiment_comparison.csv).
Every result-bearing experiment has its own directory with manifest, exact split
assignments, predictions, group metrics, and model-specific search artifacts.
Diagnostic runs are retained for auditability but excluded from comparisons.

The dataset is evaluated with grouped folds, not one permanent random split.
Traditional spectral models use fully nested parameter selection. Deep models
provide two explicit protocols: `development` selects once on a development
group, while `nested-small-data` tunes independently inside every outer fold and
reports seed ensembles. Exp-004 uses the latter. Neither protocol replaces an
untouched measurement campaign, which is not present in the repository.
