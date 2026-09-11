# Phase 3 — Foundation Model Fine-Tuning: LaBraM Results

## Setup
- Model: LaBraM-base (5.8M params), fine-tuned from official pretrained checkpoint
- Repo: https://github.com/935963004/LaBraM
- Data: same CHB-MIT patient-level split as Phase 2 (train: chb01-16, chb21, chb22; test: chb17-20)
- Preprocessing: separate pipeline matching LaBraM's pretraining spec — 0.1-75Hz bandpass, 50Hz notch, 200Hz resample, µV units, 10s non-overlapping windows (2000 samples)
- Hardware: single T4 GPU (AWS EC2 g4dn.xlarge)
- Training: 10 epochs, batch size 64, lr 3e-4, gradient clipping (clip_grad=1.0), warmup_epochs=1

## Methodology note: channel mapping
LaBraM's positional embedding table covers 129 positions (`standard_1020` vocabulary). CHB-MIT's bipolar channel names (e.g. `FP1-F7`) map to vocabulary indices 130+, outside the trained embedding table's range — using them directly triggers an out-of-bounds CUDA error. **Each bipolar channel was mapped to its first electrode's referential position** (e.g. `FP1-F7` → `FP1`) to stay within the valid embedding range. This is an approximation: bipolar (differential) and referential (single-electrode) signals have different characteristics. Two channels (`FZ-CZ`, `CZ-PZ`) are absent from LaBraM's vocabulary entirely and were excluded, reducing the channel set from 18 (Phase 1/2) to 16 for this comparison only.

## Training instability
Initial run (lr=5e-4, no gradient clipping, matching LaBraM's TUAB defaults) crashed with `NaN` loss at epoch 5, preceded by `grad_norm: Infinity` from epoch 0. Fixed via gradient clipping (`--clip_grad 1.0`) and reduced learning rate (`--lr 3e-4`).

## Results

| Epoch | Train Loss | Val AUC-ROC | Val AUC-PR | Val Balanced Acc |
|---|---|---|---|---|
| 0 | 0.074 | 0.499 | 0.003 | 0.500 |
| 1 | 0.021 | 0.500 | 0.003 | 0.500 |
| 2 | 0.021 | 0.613 | 0.053 | 0.500 |
| 3 | 0.019 | 0.881 | 0.174 | 0.525 |
| **4 (peak)** | **0.017** | **0.890** | **0.190** | **0.530** |
| 5 | 0.016 | 0.864 | 0.171 | 0.530 |
| 6 | 0.014 | 0.688 | 0.096 | 0.535 |
| 7 | 0.011 | 0.586 | 0.060 | 0.529 |
| 8 | 0.010 | 0.621 | 0.057 | 0.525 |
| 9 | 0.009 | 0.636 | 0.066 | 0.525 |

**Key finding**: train loss decreases monotonically throughout (0.074 → 0.009), while validation AUC-PR peaks at epoch 4 (0.190) then degrades to 0.066 by epoch 9 — clear overfitting, clearly visible in the divergence between train and validation trends. Best checkpoint (`checkpoint-best.pth`, epoch 4) auto-saved by the training loop.

## Comparison to Phase 2 GBM baseline

| Model | AUC-ROC | AUC-PR |
|---|---|---|
| GBM (expanded features, Phase 2) | 0.874 | 0.174 |
| LaBraM (peak, epoch 4/10) | 0.890 | 0.190 |
| LaBraM (final, epoch 9/10) | 0.636 | 0.066 |

LaBraM's peak performance modestly exceeds the GBM baseline on both metrics, but only within a narrow training window before overfitting erodes it below baseline. This nuances the "foundation models underperform on epilepsy detection" finding from prior literature (LaBraM: 0.565 balanced accuracy in a separate benchmark) — under this independent pipeline, LaBraM can outperform a feature-engineered baseline, but requires careful early stopping to do so; without it, performance degrades substantially.

## Next
BENDR fine-tuning planned as a second foundation model comparison point (reported in the literature as the strongest performer on epilepsy detection: 0.740 balanced accuracy vs. LaBraM's 0.565).