# Clinical EEG Foundation Model Probe (CHB-MIT Seizure Detection)

## Overview
Seizure detection pipeline on the CHB-MIT scalp EEG dataset (21 of 22 patients used; chb12 excluded due to a data pipeline issue). Benchmarks a feature-engineered gradient-boosted baseline against fine-tuned EEG foundation models (LaBraM complete, BENDR planned) to test whether reported foundation-model reliability issues in clinical/epilepsy tasks replicate under an independently built pipeline.

## Dataset
- **Source**: [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet)
- **Patients used**: 21 of 22 (chb12 excluded)
- **Split**: patient-level train/test (chb01-16, chb21, chb22 train; chb17-20 test) — chb21 kept with chb01 (same underlying patient, repeat recording) to avoid leakage

## Results Summary

| Model | AUC-ROC | AUC-PR | Notes |
|---|---|---|---|
| GBM baseline (126 features) | 0.870 | 0.150 | spectral + time-domain only |
| GBM + SMOTE (0.1/0.3) | 0.880-0.882 | 0.157-0.167 | shifts sensitivity/FA tradeoff, doesn't raise AUC-PR ceiling |
| **GBM (220 features, selected baseline)** | **0.874** | **0.174** | adds wavelet + cross-channel correlation features |
| LaBraM (peak, epoch 4/10) | 0.890 | 0.190 | best foundation-model result; overfits after |
| LaBraM (final, epoch 9/10) | 0.636 | 0.066 | shown for contrast — overfitting |

Full details: [`docs/phase2_results.md`](docs/phase2_results.md), [`docs/phase3_results.md`](docs/phase3_results.md)

## Pipeline

### Phase 1 — Preprocessing
- Bandpass filter (0.5-40Hz), 2s windows (non-overlapping full-recording sweep + 1s-stride overlap in seizure intervals), 18-channel canonical bipolar montage
- Memory-safe, incremental (memmap-based) processing
- Output: S3 (`processed/windows/`)

### Phase 2 — GBM Baseline
- 220 features/window: band power (5 bands) + variance + line length + wavelet energy + cross-channel correlation, per channel
- LightGBM, evaluated on held-out patients
- **Key finding**: richer features improved AUC-PR more than class-imbalance techniques (SMOTE, class weighting)

### Phase 3 — Foundation Model Benchmarking (in progress)
- **LaBraM**: complete. Separate preprocessing (0.1-75Hz filter, 50Hz notch, 200Hz resample, 16-channel subset, 10s windows). Peak result exceeds GBM baseline at epoch 4, then overfits.
- **BENDR**: planned — reported strongest performer on epilepsy detection in prior literature, second comparison point.

## Repo Structure

```
scripts/ # preprocessing, feature extraction, training scripts
data/ # local scratch (gitignored — persistent storage in S3)
docs/ # phase results, comparison tables
```


## Infrastructure
- AWS S3: `raw/`, `processed/windows/`, `processed/features/`, `processed/features_v2/`, `processed/labram/`
- AWS EC2: CPU instance (preprocessing, GBM training), GPU instance (`g4dn.xlarge`, foundation model fine-tuning)
- `uv` for Python dependency management

## Status
- ✅ Phase 1 (preprocessing) complete
- ✅ Phase 2 (GBM baseline) complete
- 🔄 Phase 3 (foundation model fine-tuning) — LaBraM complete, BENDR planned