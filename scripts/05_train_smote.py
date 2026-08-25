"""
Phase 2, Result 2: GBM baseline + SMOTE oversampling.
Same train/test split and features as train_baseline.py, for direct comparison.
"""
import os
import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, confusion_matrix
)
import boto3
from dotenv import load_dotenv

load_dotenv()

TRAIN_PATIENTS = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07",
                   "chb08", "chb09", "chb10", "chb11", "chb13", "chb14", "chb15",
                   "chb16", "chb21", "chb22"]  # chb12 excluded — data pipeline issue
TEST_PATIENTS = ["chb17", "chb18", "chb19", "chb20"]

FEATURES_DIR = "data/processed/features"
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

print("Applying SMOTE to training set...")
smote = SMOTE(random_state=42, sampling_strategy=0.3)  # minority -> 30% of majority
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_smote.shape[0]} windows, "
      f"{y_train_smote.sum()} seizure ({y_train_smote.mean()*100:.2f}%)")

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

print("\nTraining on SMOTE-resampled data...")
model.fit(X_train_smote, y_train_smote)

print("\nEvaluating on test set (untouched, natural imbalance)...")
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


# *** OUTPUT ***
# === Results (threshold=0.5 + sampling strategy=0.1) ===
# AUC-ROC: 0.8804
# AUC-PR: 0.1572
# Sensitivity (recall): 0.1447
# Precision: 0.2711
# False alarms/hour: 3.60
# Confusion matrix: TN=166948, FP=336, FN=739, TP=125

# === Threshold sweep ===
#  Threshold  Sensitivity  Precision    FA/hour     TP     FP     FN
#       0.05       0.5266     0.0378     123.84    455  11569    409
#       0.10       0.4155     0.0552      65.72    359   6139    505
#       0.15       0.3299     0.0703      40.36    285   3770    579
#       0.20       0.2778     0.0865      27.12    240   2533    624
#       0.25       0.2431     0.1111      17.99    210   1681    654
#       0.30       0.2153     0.1356      12.70    186   1186    678
#       0.35       0.1979     0.1630       9.40    171    878    693
#       0.40       0.1771     0.1952       6.75    153    631    711
#       0.45       0.1644     0.2305       5.07    142    474    722
#       0.50       0.1447     0.2711       3.60    125    336    739
#       0.55       0.1343     0.3268       2.56    116    239    748
#       0.60       0.1273     0.3915       1.83    110    171    754
#       0.65       0.1192     0.4402       1.40    103    131    761
#       0.70       0.1146     0.5051       1.04     99     97    765
#       0.75       0.1053     0.5796       0.71     91     66    773
#       0.80       0.0984     0.6855       0.42     85     39    779
#       0.85       0.0926     0.7547       0.28     80     26    784
#       0.90       0.0799     0.8625       0.12     69     11    795

# *** OUTPUT 2 ***
# === Results (threshold=0.5 + sampling strategy=0.3) ===
# AUC-ROC: 0.8815
# AUC-PR: 0.1667
# Sensitivity (recall): 0.2049
# Precision: 0.1707
# False alarms/hour: 9.21
# Confusion matrix: TN=166424, FP=860, FN=687, TP=177

# === Threshold sweep ===
#  Threshold  Sensitivity  Precision    FA/hour     TP     FP     FN
#       0.05       0.6632     0.0294     202.18    573  18887    291
#       0.10       0.5590     0.0444     111.16    483  10384    381
#       0.15       0.4745     0.0566      73.12    410   6831    454
#       0.20       0.4062     0.0681      51.45    351   4806    513
#       0.25       0.3634     0.0828      37.22    314   3477    550
#       0.30       0.3113     0.0937      27.84    269   2601    595
#       0.35       0.2766     0.1093      20.85    239   1948    625
#       0.40       0.2465     0.1251      15.94    213   1489    651
#       0.45       0.2269     0.1443      12.44    196   1162    668
#       0.50       0.2049     0.1707       9.21    177    860    687
#       0.55       0.1840     0.1941       7.07    159    660    705
#       0.60       0.1748     0.2334       5.31    151    496    713
#       0.65       0.1609     0.2860       3.71    139    347    725
#       0.70       0.1377     0.3400       2.47    119    231    745
#       0.75       0.1250     0.3942       1.78    108    166    756
#       0.80       0.1100     0.4822       1.09     95    102    769
#       0.85       0.1030     0.6268       0.57     89     53    775
#       0.90       0.0926     0.7547       0.28     80     26    784
