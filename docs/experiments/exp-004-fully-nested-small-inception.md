# Experiment exp-004-fully-nested-small-inception

## Status and hypothesis

Status: completed. This existing-data fallback tested whether the poor exp-003c
result was primarily caused by excessive capacity, globally reused
hyperparameters, single-seed variance, or lack of conservative augmentation.

The run used a low-capacity Inception model, tuned independently inside every
outer group fold. Five paired seeds compared the same selected hyperparameters
without and with fit-only augmentation. The command was:

```bash
uv run qepas-train \
  --run-id exp-004-fully-nested-small-inception \
  --skip-traditional \
  --skip-xgb-tune \
  --deep-protocol nested-small-data \
  --deep-seeds 5 \
  --deep-augmentation-ablation \
  --deep-tuner-trials 3 \
  --deep-tuner-epochs 20 \
  --deep-tuner-executions 1 \
  --deep-epochs 40 \
  --deep-batch-size 8 \
  --deep-early-stopping-patience 6 \
  --signal-length 2048 \
  --resampling-method polyphase
```

The completed run trained 21 tuner models and 70 final models in 32 minutes 48
seconds on an RTX 4070. All seven outer-fold prediction blocks were persisted
with protocol signatures so the exact run can resume safely.

## Leakage-safe split protocol

There is no permanent random train/validation/test split. Each of seven outer
folds has 100 fit scans from five groups, 20 validation scans from one group,
and 20 untouched test scans from a different group. Tuning, normalization,
early stopping, and augmentation use no outer-test data. The validation and
test groups rotate, so every selected scan receives one out-of-fold prediction.

This is a fully nested estimate for the implemented selection procedure. It is
still internal validation: all seven groups come from one sequential campaign,
and no second day or campaign exists for external testing.

## Results

| Variant | Target | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Small Inception, no augmentation | 13CO2 | 0.3298 | 0.2814 | 0.8401 |
| Small Inception, no augmentation | 12CO2 | 29.8014 | 25.1443 | 0.8367 |
| Small Inception, augmented | 13CO2 | 0.2648 | 0.2081 | 0.8969 |
| Small Inception, augmented | 12CO2 | 24.4586 | 19.9200 | 0.8900 |

The augmented ensemble reduced RMSE by 19.70% for 13CO2 and 17.93% for 12CO2
relative to its paired unaugmented ensemble. It improved on exp-001 SimpleCNN
by 14.22% and 13.14%, respectively. All five paired seeds improved under
augmentation, so this is not an ensemble-average reversal.

Augmented single-seed RMSE was 0.2819 ± 0.0308 for 13CO2 (range 0.2511–0.3302)
and 25.6911 ± 1.7010 for 12CO2 (range 23.6201–27.6207). Ensembling reduced both
errors below the single-seed means. The compact nested Ridge from exp-002 still
remains substantially better: augmented Inception RMSE is 78.53% higher for
13CO2 and 84.44% higher for 12CO2.

## Group diagnosis

The zero-concentration group `12:46:00` is the hardest: augmented RMSE is 0.4971
for 13CO2 and 44.0626 for 12CO2, with equal positive bias because every true
label is zero. Those errors are about 20% of each full target range. The next
lowest group is also positively biased. At the maximum-concentration group,
augmentation cuts RMSE by 55.02% and 45.45% relative to no augmentation.

Augmentation improves 13CO2 RMSE in six of seven held groups and 12CO2 RMSE in
five of seven. The two 12CO2 regressions are modest interior-group losses of
7.88% and 6.08%; they do not outweigh the improvements at both extremes.

Every group has exactly one label pair: concentrations increase monotonically
from `[0, 0]` at `12:46:00` to `[2.4742, 221.2258]` at `13:43:08`. Group,
concentration, and acquisition time are therefore perfectly confounded. This
dataset measures interpolation/extrapolation across seven prepared levels; it
cannot separate sensor drift from concentration response.

## Hyperparameter behavior

Only three Bayesian candidates were evaluated per outer fold by design. All
selected models used two Inception modules. The three candidate templates were
selected in 2, 3, and 2 folds, respectively, demonstrating that no one template
was uniformly best. Inner validation MAE ranged from 0.053 to 0.719; the largest
values occurred when an extreme concentration group served as validation.

The conservative augmentation operates only on fit samples and preserves phase
on the unit circle. It adds one augmented copy using measured-noise-scaled
modulus perturbations, small gain/baseline changes, and small angular phase
noise. The paired variant uses the same fold-local hyperparameters and random
initialization seed; augmentation parameters were not retuned.

## Decision and next experiment

Do not promote the CNN. Exp-004 validates conservative augmentation and
five-seed ensembling, but it misses the predeclared R² 0.90 threshold by a small
margin and remains far behind compact Ridge. Increasing depth, adding a
Transformer, or widening the tuner is not justified by seven independent
conditions.

The next result-bearing experiment should use a new campaign:

1. Randomize concentration order and repeat zero/reference measurements
   throughout the run so time drift is estimable.
2. Vary 12CO2 and 13CO2 independently; the current targets are perfectly
   collinear and cannot establish isotope-ratio identifiability.
3. Freeze the current campaign for model selection and hold the new campaign
   untouched for final testing, or pre-register a campaign-level split.
4. Carry compact nested Ridge and augmented small Inception forward unchanged.
   Treat Ridge as the primary baseline and Inception as a secondary robustness
   candidate.
5. If further existing-data work is unavoidable, test a fully nested
   cross-fitted CNN-feature plus Ridge/PLS hybrid or a physics-constrained
   single-latent target. Do not run another standalone high-capacity network.

The governing next step is experimental design and independent data, not more
architecture capacity.
