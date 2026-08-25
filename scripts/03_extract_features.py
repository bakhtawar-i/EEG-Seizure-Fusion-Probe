"""
Extract hand-crafted features from preprocessed EEG windows.
Auto-downloads from S3 if not already present locally.
Reads windows via memory-mapped load (never loads full array into RAM),
computes per-window features, saves a compact feature table + labels,
uploads to S3.

Usage:
    uv run python scripts/extract_features.py chb04
"""
import os
import sys
import numpy as np
import boto3
from scipy.signal import welch
from dotenv import load_dotenv

load_dotenv()

SFREQ = 256
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}
N_CHANNELS = 18


def ensure_local(patient_id: str):
    x_path = f"data/processed/windows/{patient_id}_X.npy"
    y_path = f"data/processed/windows/{patient_id}_y.npy"

    if os.path.exists(x_path) and os.path.exists(y_path):
        return

    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    os.makedirs("data/processed/windows", exist_ok=True)

    if not os.path.exists(x_path):
        print(f"  Downloading {patient_id}_X.npy from S3...")
        s3.download_file(bucket, f"processed/windows/{patient_id}_X.npy", x_path)
    if not os.path.exists(y_path):
        print(f"  Downloading {patient_id}_y.npy from S3...")
        s3.download_file(bucket, f"processed/windows/{patient_id}_y.npy", y_path)


def band_power(freqs, psd, low, high):
    idx = (freqs >= low) & (freqs <= high)
    return np.trapezoid(psd[idx], freqs[idx]) if idx.any() else 0.0


def extract_window_features(window: np.ndarray) -> np.ndarray:
    features = []
    for ch in range(window.shape[0]):
        sig = window[ch]
        freqs, psd = welch(sig, fs=SFREQ, nperseg=min(256, len(sig)))
        for band_name, (low, high) in BANDS.items():
            features.append(band_power(freqs, psd, low, high))
        features.append(np.var(sig))
        features.append(np.sum(np.abs(np.diff(sig))))
    return np.array(features, dtype=np.float32)


def process_patient(patient_id: str):
    ensure_local(patient_id)

    x_path = f"data/processed/windows/{patient_id}_X.npy"
    y_path = f"data/processed/windows/{patient_id}_y.npy"

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)

    n_windows = X.shape[0]
    n_features = len(BANDS) * N_CHANNELS + 2 * N_CHANNELS

    feature_matrix = np.zeros((n_windows, n_features), dtype=np.float32)

    for i in range(n_windows):
        window = np.asarray(X[i])
        feature_matrix[i] = extract_window_features(window)
        if (i + 1) % 5000 == 0:
            print(f"  {patient_id}: {i+1}/{n_windows} windows processed")

    print(f"{patient_id}: extracted {feature_matrix.shape} feature matrix, "
          f"{y.sum()} seizure windows")

    out_dir = "data/processed/features"
    os.makedirs(out_dir, exist_ok=True)
    feat_path = os.path.join(out_dir, f"{patient_id}_features.npy")
    label_path = os.path.join(out_dir, f"{patient_id}_labels.npy")
    np.save(feat_path, feature_matrix)
    np.save(label_path, y)
    print(f"Saved locally to {feat_path}, {label_path}")

    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    s3.upload_file(feat_path, bucket, f"processed/features/{patient_id}_features.npy")
    s3.upload_file(label_path, bucket, f"processed/features/{patient_id}_labels.npy")
    print(f"Uploaded to s3://{bucket}/processed/features/")

    # Clean up raw window files now that features are extracted + saved
    os.remove(x_path)
    os.remove(y_path)
    print(f"Cleaned up local {patient_id}_X.npy, {patient_id}_y.npy")


if __name__ == "__main__":
    patient_id = sys.argv[1]
    process_patient(patient_id)