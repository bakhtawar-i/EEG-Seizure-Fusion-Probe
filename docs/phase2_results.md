Phase 2 — Class Imbalance Handling: Comparison Table

Train: chb01–chb16 (excl. chb12), chb21, chb22 (17 patients) | Test: chb17–chb20 (4 patients, 20 seizure events)
Model: LightGBM, 126 spectral/time-domain features per window (band power × 5 bands + variance + line length, × 18 channels), 2s windows

Approach	AUC-ROC	AUC-PR	Sensitivity @0.5	Precision @0.5	FA/hour @0.5	TP / FP / FN @0.5
Baseline (scale_pos_weight only)	0.870	0.150	0.309	0.070	38.07	267 / 3556 / 597
SMOTE 0.1 (minority → 10% of majority)	0.880	0.157	0.145	0.271	3.60	125 / 336 / 739
SMOTE 0.3 (minority → 30% of majority)	0.882	0.167	0.205	0.171	9.21	177 / 860 / 687

Reference threshold-sweep points (Baseline vs. SMOTE 0.3), for the sensitivity/false-alarm tradeoff curve:

Threshold	Baseline Sens.	Baseline FA/hr	SMOTE 0.3 Sens.	SMOTE 0.3 FA/hr
0.05	0.699	236.6	0.663	202.2
0.20	0.505	96.8	0.406	51.5
0.50	0.309	38.1	0.205	9.2
0.90	0.159	8.4	0.093	0.28

Key finding: AUC-ROC and AUC-PR improve marginally and consistently as SMOTE ratio increases (0.870→0.882 ROC, 0.150→0.167 PR), but this reflects a shift in operating point along the same tradeoff curve, not improved class separability. AUC-PR plateaus around 0.15–0.17 across all three variants — the real ceiling. All three approaches share the same underlying limitation: even at the most permissive threshold (0.05), maximum achievable sensitivity is ~70%, meaning the current feature set (spectral band power + basic time-domain statistics) does not separate seizure/non-seizure windows cleanly enough for imbalance-handling technique alone to resolve.

Cross-patient generalization context (from literature review): patient-specific CHB-MIT detectors report sensitivity in the 90–97% range at <0.1–0.4 FA/hour, but a comparable large-scale cross-patient study (205 patients, 3 centers) reported only 67% sensitivity at 0.32 FA/hour on CHB-MIT — the harder, generalizing setup this project uses, and the more appropriate benchmark for comparison.

Next step: feature richness, not imbalance handling, is the likely limiting factor. Planned: add wavelet-domain and cross-channel/spatial features (informed by literature showing meaningful gains from multi-domain feature sets over spectral-only approaches).