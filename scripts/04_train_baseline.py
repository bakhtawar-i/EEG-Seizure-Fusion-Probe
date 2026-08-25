"""
Train GBM baseline on extracted EEG features. Patient-level train/test split.
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

FEATURES_DIR = "data/processed/features"


def ensure_local(patient_id: str):
    feat_path = f"{FEATURES_DIR}/{patient_id}_features.npy"
    label_path = f"{FEATURES_DIR}/{patient_id}_labels.npy"
    if os.path.exists(feat_path) and os.path.exists(label_path):
        return
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    os.makedirs(FEATURES_DIR, exist_ok=True)
    if not os.path.exists(feat_path):
        s3.download_file(bucket, f"processed/features/{patient_id}_features.npy", feat_path)
    if not os.path.exists(label_path):
        s3.download_file(bucket, f"processed/features/{patient_id}_labels.npy", label_path)


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

# Class weighting: scale_pos_weight = n_negative / n_positive
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

model = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

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
# false alarms per hour: FP windows * window_duration_sec / 3600
window_sec = 2
false_alarms_per_hour = (fp * window_sec) / 3600 / (len(y_test) * window_sec / 3600) * len(y_test)
# simpler: total false-positive time / total recording time in hours
total_hours = (len(y_test) * window_sec) / 3600
fa_per_hour = fp / total_hours

print(f"\n=== Results ===")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")
print(f"Sensitivity (recall): {sensitivity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"False alarms/hour: {fa_per_hour:.2f}")
print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# *** OUTPUT ***

# === Results ===
# AUC-ROC: 0.8696
# AUC-PR: 0.1501
# Sensitivity (recall): 0.3090
# Precision: 0.0698
# False alarms/hour: 38.07
# Confusion matrix: TN=163728, FP=3556, FN=597, TP=267

"""
Threshold sweep on the already-trained baseline model's predictions.
Run this after train_baseline.py has produced y_pred_proba for the test set.
"""
import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix

# Assumes y_test and y_pred_proba are available — either rerun training inline
# or save/load them. Simplest: just append this to train_baseline.py after
# the existing evaluation block, reusing y_test and y_pred_proba directly.

thresholds = np.arange(0.05, 0.95, 0.05)
window_sec = 2

print(f"{'Threshold':>10} {'Sensitivity':>12} {'Precision':>10} {'FA/hour':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    sens = recall_score(y_test, y_pred_t, zero_division=0)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    total_hours = (len(y_test) * window_sec) / 3600
    fa_per_hour = fp / total_hours
    print(f"{t:>10.2f} {sens:>12.4f} {prec:>10.4f} {fa_per_hour:>10.2f} {tp:>6} {fp:>6} {fn:>6}")

# *** OUTPUT ***

#  Threshold  Sensitivity  Precision    FA/hour     TP     FP     FN
#       0.05       0.6991     0.0266     236.58    604  22100    260
#       0.10       0.6227     0.0349     159.35    538  14886    326
#       0.15       0.5637     0.0409     122.21    487  11416    377
#       0.20       0.5046     0.0460      96.83    436   9045    428
#       0.25       0.4653     0.0507      80.50    402   7520    462
#       0.30       0.4248     0.0545      68.21    367   6372    497
#       0.35       0.3866     0.0578      58.30    334   5446    530
#       0.40       0.3507     0.0609      50.05    303   4675    561
#       0.45       0.3241     0.0644      43.52    280   4065    584
#       0.50       0.3090     0.0698      38.07    267   3556    597
#       0.55       0.2986     0.0771      33.06    258   3088    606
#       0.60       0.2812     0.0823      29.02    243   2711    621
#       0.65       0.2627     0.0869      25.54    227   2386    637
#       0.70       0.2442     0.0948      21.56    211   2014    653
#       0.75       0.2245     0.1030      18.08    194   1689    670
#       0.80       0.2025     0.1103      15.12    175   1412    689
#       0.85       0.1852     0.1258      11.90    160   1112    704
#       0.90       0.1586     0.1491       8.37    137    782    727
