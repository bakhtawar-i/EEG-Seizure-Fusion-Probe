"""
Retrain the Phase 2 expanded-features GBM baseline, save the model, and
generate per-window test predictions tagged by patient — needed for the
cross-model seizure-level error inspection.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import boto3
from dotenv import load_dotenv

load_dotenv()

TRAIN_PATIENTS = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07",
                   "chb08", "chb09", "chb10", "chb11", "chb13", "chb14", "chb15",
                   "chb16", "chb21", "chb22"]
TEST_PATIENTS = ["chb17", "chb18", "chb19", "chb20"]

FEATURES_DIR = "data/processed/features_v2"


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


def load_patients_tagged(patient_list):
    """Like load_patients, but also returns which patient each window belongs to."""
    X_list, y_list, patient_tags = [], [], []
    for p in patient_list:
        ensure_local(p)
        X = np.load(f"{FEATURES_DIR}/{p}_features.npy")
        y = np.load(f"{FEATURES_DIR}/{p}_labels.npy")
        X_list.append(X)
        y_list.append(y)
        patient_tags.extend([p] * len(y))
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), np.array(patient_tags)


print("Loading train set...")
X_train, y_train, _ = load_patients_tagged(TRAIN_PATIENTS)

print("Loading test set...")
X_test, y_test, test_patient_tags = load_patients_tagged(TEST_PATIENTS)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)

print("Training...")
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Save model
os.makedirs("saved_models", exist_ok=True)
joblib.dump(model, "saved_models/gbm_baseline.joblib")

# Save per-window test predictions, tagged by patient, for error inspection
predictions_df = pd.DataFrame({
    "patient": test_patient_tags,
    "y_true": y_test,
    "gbm_prob": y_pred_proba,
})
predictions_df.to_csv("saved_models/gbm_test_predictions.csv", index=False)

print(f"Model saved to saved_models/gbm_baseline.joblib")
print(f"Predictions saved to saved_models/gbm_test_predictions.csv ({len(predictions_df)} rows)")

# Upload both to S3
s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
bucket = os.environ["S3_BUCKET_NAME"]
s3.upload_file("saved_models/gbm_baseline.joblib", bucket, "models/gbm/gbm_baseline.joblib")
s3.upload_file("saved_models/gbm_test_predictions.csv", bucket, "models/gbm/gbm_test_predictions.csv")
print("Uploaded to S3")