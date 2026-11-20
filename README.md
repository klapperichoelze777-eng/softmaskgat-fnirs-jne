# SoftMaskGAT for fNIRS

SoftMaskGAT classifies participant-level fNIRS recordings with variable channel availability. It combines a temporal encoder, a quality-weighted Anchor-Lite residual, graph attention on channel coordinates and a weighted readout. The GAP and attention branches produce separate probabilities, which are combined with equal weights.

## Run the example

From a checkout of this repository, use Python 3.10 or later with PyTorch, NumPy, SciPy and scikit-learn.

```sh
python -m pip install .
python -m softmaskgat demo
python -m softmaskgat tables
```

The first command after installation fits GAP and Attn on synthetic HbO/HbR signals. It uses eight simulated participants, CPU, one epoch per fold, one seed and 128 bootstrap draws. The held-out probability array has shape `(1, 1, 2, 8)` for one condition, one seed, two branches and eight participants. The example prints the branch and fusion AUCs. Synthetic labels and signals are generated locally. The tables command reads three bundled group-level CSVs and reconstructs the cells of tables 2, 11 and 13 at the paper’s reporting precision. These commands can be run from any directory after installation.

## Compare channel handling and quality weighting

```sh
python -m softmaskgat demo --protocol handling
python -m softmaskgat demo --protocol core
python -m softmaskgat demo --protocol targeted
python -m softmaskgat demo --protocol quality
```

| Protocol | Study evaluation | Output and conditions |
|---|---|---|
| primary | Participant-level LOSO, three seeds (42, 123, 2026) | GAP and Attn probabilities averaged within branch, then fixed 50/50 fusion. The original analysis fits the label-free quality normalisation across the cohort. |
| handling | LOSO, seed 42 | Full topology with fold-wise quality, induced-subgraph deletion (D1A) and spatial interpolation (I2). Tables 9 and 10 give fused performance and paired differences. |
| core | One stratified ten-fold partition with three optimisation streams | FULL, binary weights with the finite attention floor (B_floor), D1A, I2 and a learned quality exponent (SHRINK). Table 11 reports prespecified paired AUC differences. |
| targeted | GAP, LOSO, seed 42 | Binary attention gating, equal-weight valid-node readout, binary-only propagation and removal of the Anchor-Lite residual, each compared with FULL in table 12. |
| quality | Three stratified ten-fold partitions | FULL, B_floor, participant-mean or permuted quality, readout-only quality, quality excluded from attention, and matched scalar-input conditions. Table 13 reports selected contrasts from the full nine-comparison family. |

The additional protocols fit quality-normalisation parameters within the outer training fold. In the ten-fold comparisons, probabilities are fused within each run before computing the paired AUC difference. Those differences are then averaged over runs. The demo uses two stratified folds for the ten-fold examples. The paper settings are the defaults for `run`.

## Use a processed dataset

Supply an NPZ file containing the following aligned numeric arrays. Each participant occupies one row.

| Array | Shape | Meaning |
|---|---|---|
| signal | N × 991 × 48 × 2 | Model-ready, baseline-corrected and resampled HbO then HbR at 11 Hz. The first 55 samples are baseline and the next 660 are the task. |
| mask | N × 48 | Binary channel availability. The loader zero-fills unavailable signal and Anchor-Lite entries. |
| label | N | Binary outcome, with BD = 1 and control = 0 for the study task. |
| ids | N | Distinct participant identifiers in a numeric or fixed-width string array. |
| coords | 48 × 3 | Ordered channel midpoints for the 3-D graph. I2 uses the first two coordinate dimensions as the probe plane. |
| anchor (optional) | N × 48 × 16 | Precomputed raw Anchor-Lite values. If omitted, the loader computes the 16 values from the processed signals. |

```sh
python -m softmaskgat run study.npz --protocol primary --device cuda --output outputs/primary
python -m softmaskgat run study.npz --protocol core --device cuda --output outputs/core
python -m softmaskgat tables --output outputs/paper-tables
```

`run` uses a maximum of 120 epochs, a batch size of 16 and training-loss early stopping with patience 30. CUDA enables mixed precision. A new output directory receives settings, participant-aligned held-out probabilities and statistics. The axes of the probability array are condition, run, branch and participant. Full study protocols require substantially more computation than the synthetic example. The original clinical data follow the data availability statement in the article.

The primary and seed-42 paired analyses use ordinary participant bootstrap resampling. The primary analysis redraws a single-class sample. The handling and targeted comparisons retain all draws for BAcc, MCC and Brier, and record the number of two-class draws used for AUC. The fixed ten-fold and repeated ten-fold comparisons resample within diagnostic group with shared participant draws across conditions. Estimates from multiple runs are combined according to the protocol above. For an alternative input to the tables command, use `--supplement PATH` with the study aggregate ZIP containing `r3/ensemble_interval_comparison.csv`, `core_families.csv` and `q_comparisons.csv`.

## Citation and maintenance

Chen K, Wu X, He L and Meng Y. *Quality-aware mask-propagation graph attention for fNIRS-based bipolar disorder classification without channel interpolation*. Journal of Neural Engineering (2026). [Article DOI](https://doi.org/10.1088/1741-2552/aebe6b)

Please cite the article when using the method. Cite the archived software when referring to this implementation. [SoftMaskGAT v1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.24394153)

Maintained by [Jack Chen](https://github.com/klapperichoelze777-eng). Source code is distributed under the MIT license.
