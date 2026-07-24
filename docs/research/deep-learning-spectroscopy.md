# Deep-learning architectures for QEPAS spectral regression

Research date: 2026-07-24. The review prioritizes primary papers on quantitative
one-dimensional spectroscopy, photoacoustic gas sensing, and closely related
sequence architectures.

## Evidence map

| Architecture / paper | Reported use | Project decision |
|---|---|---|
| [DeepSpectra](https://doi.org/10.1016/j.aca.2019.01.002) | End-to-end quantitative Vis/NIR calibration using parallel Inception filters on raw spectra; the paper reports that accuracy and repeatability improve with sample count. | Implemented as `inception_spectra`: bottleneck 1x1 convolutions, three receptive fields, max-pool branch, optional residual links, and global average pooling. |
| [InceptionTime](https://doi.org/10.1007/s10618-020-00710-y) | Six stacked multi-scale Inception modules, bottlenecks, residual blocks, global average pooling, and a five-seed ensemble for time-series classification. | Structural motifs and a five-seed regression ensemble are implemented. Exp-004 confirms that ensembling reduces error but does not close the compact-Ridge gap. |
| [Cui and Fearn](https://doi.org/10.1016/j.chemolab.2018.07.008) | Unified 1-D CNN calibration compared with PLSR on NIR datasets with 6,998, 1,000, and 415 training spectra. | Supports the existing `simple_cnn`/`multiscale_cnn`, but its sample scale is materially larger than this campaign's 140 selected scans and seven independent label groups. |
| [Malek, Melgani, and Bazi](https://doi.org/10.1002/cem.2977) | 1-D CNN feature extraction followed by SVR or Gaussian-process regression on three spectroscopic datasets. | A useful future hybrid: freeze or cross-fit learned filters, then use a low-variance regressor. Not run yet because feature learning must remain inside each outer fold. |
| [Photoacoustic residual network](https://doi.org/10.1016/j.pacs.2024.100647) | A 40-layer residual network for methane concentration retrieval and noise suppression in photoacoustic spectroscopy. | Added `dilated_resnet1d`, a much smaller non-causal residual model with exponentially varying receptive fields. Forty layers are not justified by this dataset size. |
| [EMD-CNN-LSTM mixed-gas PAS](https://doi.org/10.1021/acs.analchem.4c04479) | EMD plus CNN-LSTM on overlapping WMS-2f signals; 25 concentration combinations, cyclic-shift/noise augmentation, and five extra measurement sets. | CNN/LSTM support already exists. Exp-004 implements fit-only measured-noise, gain, baseline, and angular-phase perturbations; spectral shifts and EMD remain deferred until drift is measured. |
| [Direct-absorption gas retrieval](https://doi.org/10.1016/j.measurement.2021.109739) | 1-D CNN and MLP with simulated noisy spectra and transfer learning for methane/acetylene retrieval. | Simulation pretraining is higher priority than a larger Transformer if an instrument-faithful forward model can be built. Synthetic and measured data must be tracked separately. |

## Implemented architecture registry

The registry now exposes eight common-contract models:

- `simple_cnn`
- `resnet1d`
- `tcn`
- `lstm`
- `multiscale_cnn`
- `transformer1d`
- `inception_spectra`
- `dilated_resnet1d`

Every builder accepts the same signal/scalar inputs and produces the two
concentration outputs. Architecture-specific hyperparameters are conditional,
so a dilated-network trial does not carry inactive Inception dimensions.

The search space now covers depth, filters, receptive field, downsampling,
residual use, scalar/head widths, batch/layer/group/no normalization, activation,
dropout, L2, Adam/AdamW, weight decay, gradient clipping, constant/exponential/
cosine/restart schedules, warm-up, MSE/MAE/Huber/LogCosh loss, and repeated
initializations per trial. Trials are ranked by normalized-target validation MAE,
which remains comparable when their optimized loss functions differ.

## Applicability to this campaign

There are 140 selected scans but only seven independently held concentration/time
groups and seven distinct paired labels. Replicate scans are not equivalent to
140 independent concentration conditions. That makes model variance and
concentration-time confounding the governing constraints.

The literature often succeeds with hundreds or thousands of calibration spectra,
simulation pretraining, explicit augmentation, or separate-condition tests.
Exp-003c confirmed the sample-size warning: one broad Inception trial reached
normalized validation MAE 0.186 on a single development group, yet grouped R²
fell to 0.631 for 13CO2 and 0.542 for 12CO2. Exp-004 then nested a constrained
two-module Inception search inside every outer fold and averaged five seeds.
Fit-only augmentation reduced RMSE by 19.70% and 17.93%, reaching R² 0.897 and
0.890, but compact nested Ridge remained markedly better at R² 0.968.

## Priority order

1. Use nested compact Ridge/PLS as the current scientific baseline.
2. Acquire an independent campaign with randomized concentration order,
   repeated zero/reference measurements, and independently varied isotope
   ratios.
3. Carry the exp-004 fit-only augmentation and five-seed ensemble forward as
   validated deep-learning controls; do not increase capacity on this campaign.
4. If existing-data work is unavoidable, prefer a fully nested cross-fitted
   CNN-feature plus Ridge/PLS hybrid or a physically constrained single-latent
   target.
5. Prefer simulation pretraining only after an instrument-faithful forward
   model and synthetic/measured provenance boundary exist.
6. Defer Transformers, deeper residual networks, spectral shifts, and
   EMD-CNN-LSTM until there are more independent conditions or measured drift.

The architecture registry makes those experiments possible, but architecture
availability should not be mistaken for evidence that a high-capacity model is
appropriate for the current data.
