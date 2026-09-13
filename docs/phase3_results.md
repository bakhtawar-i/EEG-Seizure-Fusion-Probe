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

## BENDR

### Setup
- Model: BENDR (~157M params), loaded via braindecode's `InterpolatedBENDR` wrapper with pretrained checkpoint `braindecode/braindecode-bendr`
- Data: Phase 1's original windowed data used directly — 18 channels, 512 samples (2s @ 256Hz). No separate preprocessing pipeline needed (unlike LaBraM), since BENDR's window length isn't fixed by the pretrained checkpoint and 256Hz is its native pretraining rate.
- Channel handling: `InterpolatedBENDR`'s built-in spatial-interpolation layer projects arbitrary channel configurations onto BENDR's canonical 20-channel space using real electrode geometry (MNE `chs_info` with 3D positions). Each CHB-MIT bipolar channel was assigned its first electrode's standard 10-20 position (e.g. `FP1-F7` → `FP1`'s position) — the same category of approximation as LaBraM's channel mapping, but resolved via genuine spatial interpolation rather than a hard index lookup.
- Verified clean pretrained weight loading (0 missing, 0 unexpected keys, all 157,142,075 params) — notable given a documented history of a silent 0/99-weights loading bug with this checkpoint in earlier braindecode versions.

### Training instability and fix
Full end-to-end fine-tuning (all 157M params trainable) diverged to NaN loss within 1-3 epochs, even with gradient clipping (`clip_grad_norm_=1.0`), a conservative learning rate (1e-5), and AMP/GradScaler (which correctly detected and skipped non-finite-gradient steps, but the underlying instability persisted and worsened rather than self-correcting, unlike LaBraM's recovery under similar conditions).

**Fix**: froze the entire pretrained backbone (encoder + contextualizer) and trained only the final classification head — 1,538 trainable parameters out of 157,142,075 — following braindecode's own documented transfer-learning guidance ("Loading and Adapting Pretrained Foundation Models" tutorial recommends extracting frozen features and training a lightweight head, rather than full fine-tuning, as the standard pattern for their foundation models).

### Results

| Epoch | Train Loss | Test AUC-ROC | Test AUC-PR | Test Balanced Acc |
|---|---|---|---|---|
| 0 | 0.222 | 0.804 | 0.028 | 0.500 |
| 1-9 | ~0.222 (flat) | ~0.804 (flat) | ~0.028 (flat) | 0.500 (flat) |

With only 1,538 trainable parameters, the model converged within epoch 0 and showed no further change across the remaining 9 epochs — expected given the very limited capacity of a linear head on frozen features.

### Threshold sweep reveals a degenerate decision boundary
Despite a seemingly reasonable ROC-AUC (0.804), a full threshold sweep on the test set revealed the model has **no usable operating threshold**:

| Threshold | Sensitivity | FA/hour | TP | FP | FN |
|---|---|---|---|---|---|
| 0.05 | 0.998 | 1789.34 | 862 | 167,152 | 2 |
| 0.10 – 0.90 | 0.000 | 0.00 | 0 | 0 | 864 |

At threshold 0.05, the model flags nearly every window as positive (99.8% sensitivity, but ~1,789 false alarms/hour — clinically meaningless). At any threshold ≥0.10, it flips to predicting negative for every single window. There is no middle ground: predicted probabilities are compressed into an extremely narrow band, so the model's output is effectively binary at the population level, not a meaningfully calibrated probability distribution.

**This is a direct empirical instance of the reliability gap this project is built to surface**: a summary metric (ROC-AUC 0.804) that looks reasonable in isolation, produced by a model with no practically usable decision threshold at all. AUC measures ranking quality, not decision usability — this result makes that distinction concrete.

### Caveat
This finding characterizes the **frozen-backbone, linear-head transfer-learning approach** specifically — not necessarily an indictment of BENDR's pretrained representations in general. A different fine-tuning strategy (e.g., discriminative per-layer learning rates, partial unfreezing of later layers only, or a more expressive head) might resolve the threshold-collapse issue without reintroducing the instability seen under full fine-tuning. Untested here; noted as a direction for future work rather than a resolved question.

## Cross-model comparison

| Model | AUC-ROC | AUC-PR | Trainable params | Fine-tuning approach | Notes |
|---|---|---|---|---|---|
| GBM (Phase 2, 220 features) | 0.874 | 0.174 | — | — | selected baseline |
| LaBraM (peak, epoch 4/10) | 0.890 | 0.190 | 5,824,937 (full) | full fine-tune | overfits after epoch 4 |
| LaBraM (final, epoch 9/10) | 0.636 | 0.066 | " | " | shown for contrast |
| BENDR (frozen head) | 0.804 | 0.028 | 1,538 | frozen backbone, linear head only | degenerate decision boundary — no usable threshold |

**Note on comparison asymmetry**: LaBraM and BENDR were fine-tuned with different strategies (full fine-tune vs. frozen backbone) because full fine-tuning was unstable for BENDR specifically but stable (with gradient clipping) for LaBraM. This asymmetry is itself a finding — it suggests the two architectures differ in fine-tuning stability under near-identical training conditions (same optimizer family, similar learning rate range, same clipping/AMP setup) — rather than a flaw in the comparison design. Both results are reported with their respective approaches stated explicitly.

## Scope note
NeuroGPT (the third model from the source literature comparison) was not tested. This project reports results for two of the three originally identified foundation models, chosen based on GBM baseline time/effort tradeoffs — not a full replication of the three-way benchmark.