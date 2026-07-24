# Package architecture

## Dependency direction

```text
interfaces ──> application ──> data ──> features
                    │          │
                    ├────────> models ──> validation
                    │             │
                    ├────────> evaluation
                    └────────> visualization

all layers ──> core
```

The arrows indicate allowed implementation dependencies. The CLI parses user
input and constructs `TrainingOptions`; it does not assemble datasets, train
models, or write experiment artifacts itself.

## Package responsibilities

- `core`: stable configuration and reproducibility values.
- `data`: scan discovery, validation, dataset assembly, normalization, and profiling.
- `features`: deterministic transformations from one scan to model inputs.
- `validation`: pure grouped splitting policies shared by model families.
- `models/traditional`: sklearn estimator factories, grouped evaluation, and tuning.
- `models/deep_learning`: Keras builders, schedules, tuning, and grouped folds.
- `evaluation`: result value objects, metric computation, and persistence.
- `visualization`: rendering only; no training decisions.
- `application`: the `TrainingPipeline` use case and its typed command objects.
- `interfaces`: Typer or future HTTP/notebook adapters.

## Canonical imports

```python
from qepas_spectroscopy.application import TrainingOptions, TrainingPipeline
from qepas_spectroscopy.data import DatasetPipeline, NormalizationBundle, Scan
from qepas_spectroscopy.evaluation import compute_metrics
from qepas_spectroscopy.models.traditional import build_ridge
from qepas_spectroscopy.models.deep_learning import train_deep_model
from qepas_spectroscopy.validation import nested_group_folds
```

Legacy mirror modules were removed after the reorganization. Every
responsibility now has one canonical import path, preventing two module trees
from drifting apart.

## Adding functionality

- Add a new data source behind the `DatasetPipeline` dependencies or a sibling
  builder in `data/`.
- Add a traditional estimator factory in `models/traditional/factories.py`.
- Add a Keras architecture in `models/deep_learning/architectures.py` and
  register it in `hypermodel.py`; the architecture contract test must pass.
- Add another interface by translating its inputs into `TrainingOptions` and
  invoking `TrainingPipeline`.
- Keep scientific split and preprocessing decisions out of interfaces and plots.

## Data flow

```text
scan folders
    -> discovery and schema validation
    -> one-pass DatasetPipeline
       -> engineered features -> fold-local Ridge scaling or tree models
       -> polyphase resampling + circular phase encoding
          -> optional fit-only physics-conservative augmentation
          -> fold-local signal/scalar/target normalization -> deep models
```

Preprocessing is deterministic and label-independent. Any fitted state belongs to
the training partition and is persisted with a preprocessing fingerprint.
