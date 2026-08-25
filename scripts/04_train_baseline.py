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

TRAIN_PATIENTS = [f"chb{str(i).zfill(2)}" for i in range(1, 17)] + ["chb21", "chb22"]
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