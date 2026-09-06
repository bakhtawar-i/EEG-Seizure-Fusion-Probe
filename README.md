# Clinical EEG Foundation Model Probe (CHB-MIT Seizure Detection)

## Overview
Seizure detection pipeline on the CHB-MIT scalp EEG dataset (21 patients used, chb12 excluded due to a data pipeline issue). Benchmarks a feature-engineered gradient-boosted baseline against fine-tuned EEG foundation models (LaBraM, BENDR, NeuroGPT) to test whether reported foundation-model reliability issues in clinical/epilepsy tasks replicate under an independently built pipeline.

## Dataset
- **Source**: [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet)
- **Patients used**: 21 of 22 (chb12 excluded — raw data lost during a disk-cleanup step, not reprocessed)
- **Split**: patient-level train/test (chb01–16, chb21, chb22 train; chb17–20 test) — chb21 kept with chb01 (same underlying patient, repeat recording) to avoid leakage

## Pipeline

### Phase 1 — Preprocessing
- Bandpass filter (0.5–40Hz), 2-second windows, non-overlapping across full recordings with 1s-stride overlap in seizure intervals (+2s margin) to densify sparse positive samples
- 18-channel canonical set (bipolar montage), standardized across all patients after discovering channel inconsistencies in the raw dataset
- Memory-safe, incremental (memmap-based) processing to handle large patients on limited RAM
- All processed data stored in S3 (`processed/windows/`)

### Phase 2 — GBM Baseline
- 220 hand-crafted features per window: band power (5 bands) + variance + line length per channel, plus wavelet-domain energy and cross-channel correlation features
- LightGBM classifier, evaluated on held-out patients
- **Result**: AUC-ROC 0.874, AUC-PR 0.174, sensitivity 0.335 @ 34.5 false alarms/hour (threshold=0.5)
- **Key finding**: richer features (specifically cross-channel correlation) improved AUC-PR more than class-imbalance techniques (SMOTE, class weighting) — full comparison table in `docs/phase2_results.md`

### Phase 3 — Foundation Model Benchmarking (in progress)
- Separate preprocessing pipeline matching each model's pretraining spec (e.g., LaBraM: 0.1–75Hz filter, 50Hz notch, 200Hz resample, 16-channel subset, 10s windows)
- Fine-tuning LaBraM, BENDR, and NeuroGPT on CHB-MIT, single-GPU (AWS EC2 `g4dn.xlarge`)
- Motivated by a published benchmark finding LaBraM underperforms specifically on epilepsy detection (0.565 balanced accuracy) relative to BENDR (0.740) and NeuroGPT (0.734) — testing whether this replicates independently

## Repo Structure

```
scripts/ # preprocessing, feature extraction, training scripts
data/ # local scratch (gitignored — see S3 for persistent storage)
docs/ # phase results, comparison tables
```


## Infrastructure
- AWS S3 for data storage (`raw/`, `processed/windows/`, `processed/features/`, `processed/features_v2/`, `processed/labram/`)
- AWS EC2 for compute: CPU instance (preprocessing, GBM training), GPU instance (`g4dn.xlarge`, foundation model fine-tuning)
- `uv` for Python dependency management

## Status
- ✅ Phase 1 (preprocessing) complete
- ✅ Phase 2 (GBM baseline) complete
- 🔄 Phase 3 (foundation model fine-tuning) in progress — LaBraM preprocessing underway
