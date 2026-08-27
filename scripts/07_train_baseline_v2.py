"""
Phase 2, Result 3: GBM baseline on expanded feature set (wavelet + cross-channel).
Same train/test split as train_baseline.py, for direct comparison.
"""
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, confusion_matrix
)
import boto3
from dotenv import load_dotenv

load_dotenv()

TRAIN_PATIENTS = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07",
                   "chb08", "chb09", "chb10", "chb11", "chb13", "chb14", "chb15",
                   "chb16", "chb21", "chb22"]
TEST_PATIENTS = ["chb17", "chb18", "chb19", "chb20"]

FEATURES_DIR = "data/processed/features_v2"
WINDOW_SEC = 2


def ensure_local(patient_id: str):
    feat_path = f"{FEATURES_DIR}/{patient_id}_features.npy"
    label_path = f"{FEATURES_DIR}/{patient_id}_labels.npy"
    if os.path.exists(feat_path) and os.path.exists(label_path):
        return
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    os.makedirs(FEATURES_DIR, exist_ok=True)
    if not os.path.exists(feat_path):
        s3.download_file(bucket, f"processed/features_v2/{patient_id}_features.npy", feat_path)
    if not os.path.exists(label_path):
        s3.download_file(bucket, f"processed/features_v2/{patient_id}_labels.npy", label_path)


def load_patients(patient_list):
    X_list, y_list = [], []
    for p in patient_list:
        ensure_local(p)
        X = np.load(f"{FEATURES_DIR}/{p}_features.npy")
        y = np.load(f"{FEATURES_DIR}/{p}_labels.npy")
        X_list.append(X)
        y_list.append(y)
        print(f"{p}: {X.shape[0]} windows, {y.sum()} seizure")
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


print("Loading train set...")
X_train, y_train = load_patients(TRAIN_PATIENTS)
print(f"\nTrain: {X_train.shape[0]} windows, {y_train.sum()} seizure ({y_train.mean()*100:.3f}%)\n")

print("Loading test set...")
X_test, y_test = load_patients(TEST_PATIENTS)
print(f"\nTest: {X_test.shape[0]} windows, {y_test.sum()} seizure ({y_test.mean()*100:.3f}%)\n")

# Guard against any NaN/Inf from cross-channel correlation on constant-signal windows
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

model = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=2,          # match instance vCPU count
    verbose=1,          # shows training progress
)

# model = lgb.LGBMClassifier(
#     objective="binary",
#     scale_pos_weight=scale_pos_weight,
#     n_estimators=500,
#     learning_rate=0.05,
#     num_leaves=31,
#     random_state=42,
# )

print("\nTraining...")
model.fit(X_train, y_train)

print("\nEvaluating on test set...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr = average_precision_score(y_test, y_pred_proba)
sensitivity = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
total_hours = (len(y_test) * WINDOW_SEC) / 3600
fa_per_hour = fp / total_hours

print(f"\n=== Results (threshold=0.5) ===")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")
print(f"Sensitivity (recall): {sensitivity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"False alarms/hour: {fa_per_hour:.2f}")
print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print(f"\n=== Threshold sweep ===")
print(f"{'Threshold':>10} {'Sensitivity':>12} {'Precision':>10} {'FA/hour':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
for t in np.arange(0.05, 0.95, 0.05):
    y_pred_t = (y_pred_proba >= t).astype(int)
    sens = recall_score(y_test, y_pred_t, zero_division=0)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
    fa_t = fp_t / total_hours
    print(f"{t:>10.2f} {sens:>12.4f} {prec:>10.4f} {fa_t:>10.2f} {tp_t:>6} {fp_t:>6} {fn_t:>6}")

# Feature importance — useful for the feature-selection step literature suggested
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:20]
print(f"\n=== Top 20 features by importance (index) ===")
for idx in top_idx:
    print(f"Feature {idx}: importance {importances[idx]}")