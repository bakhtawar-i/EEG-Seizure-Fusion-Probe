"""
Phase 2, Result 3: expanded feature extraction — adds wavelet-domain and
cross-channel correlation features to the original spectral/time-domain set.

Original 126 features: band power (5 bands) + variance + line length, x18 channels
New: wavelet energy (5 levels) x18 channels = 90 features
New: cross-channel correlation summary (mean/std/max/min of pairwise
     correlation across all channel pairs) = 4 features
Total: 220 features per window

Usage:
    uv run python scripts/06_extract_features_v2.py chb04
"""
import os
import sys
import numpy as np
import pywt
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
WAVELET = "db4"
WAVELET_LEVELS = 4  # produces 5 coefficient arrays: 4 detail + 1 approx


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
    if not idx.any():
        return 0.0
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return trapz_fn(psd[idx], freqs[idx])


def spectral_time_features(sig: np.ndarray) -> list:
    """Original 7 features per channel: 5 band powers + variance + line length."""
    freqs, psd = welch(sig, fs=SFREQ, nperseg=min(256, len(sig)))
    feats = [band_power(freqs, psd, low, high) for low, high in BANDS.values()]
    feats.append(np.var(sig))
    feats.append(np.sum(np.abs(np.diff(sig))))
    return feats


def wavelet_features(sig: np.ndarray) -> list:
    """5 features per channel: energy at each decomposition level."""
    coeffs = pywt.wavedec(sig, WAVELET, level=WAVELET_LEVELS)
    return [np.sum(np.square(c)) for c in coeffs]  # [cA4, cD4, cD3, cD2, cD1] energies


def cross_channel_features(window: np.ndarray) -> list:
    """4 global features: mean/std/max/min of pairwise channel correlation."""
    corr_matrix = np.corrcoef(window)  # (n_channels, n_channels)
    iu = np.triu_indices_from(corr_matrix, k=1)
    pairwise = corr_matrix[iu]
    pairwise = pairwise[~np.isnan(pairwise)]
    if len(pairwise) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [pairwise.mean(), pairwise.std(), pairwise.max(), pairwise.min()]


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """window shape: (n_channels, n_samples) -> flat feature vector, 220-dim"""
    features = []
    for ch in range(window.shape[0]):
        sig = window[ch]
        features.extend(spectral_time_features(sig))
        features.extend(wavelet_features(sig))
    features.extend(cross_channel_features(window))
    return np.array(features, dtype=np.float32)


def process_patient(patient_id: str):
    ensure_local(patient_id)

    x_path = f"data/processed/windows/{patient_id}_X.npy"
    y_path = f"data/processed/windows/{patient_id}_y.npy"

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)

    n_windows = X.shape[0]
    n_features = N_CHANNELS * (len(BANDS) + 2 + (WAVELET_LEVELS + 1)) + 4

    feature_matrix = np.zeros((n_windows, n_features), dtype=np.float32)

    for i in range(n_windows):
        window = np.asarray(X[i])
        feature_matrix[i] = extract_window_features(window)
        if (i + 1) % 5000 == 0:
            print(f"  {patient_id}: {i+1}/{n_windows} windows processed")

    print(f"{patient_id}: extracted {feature_matrix.shape} feature matrix, "
          f"{y.sum()} seizure windows")

    out_dir = "data/processed/features_v2"
    os.makedirs(out_dir, exist_ok=True)
    feat_path = os.path.join(out_dir, f"{patient_id}_features.npy")
    label_path = os.path.join(out_dir, f"{patient_id}_labels.npy")
    np.save(feat_path, feature_matrix)
    np.save(label_path, y)
    print(f"Saved locally to {feat_path}, {label_path}")

    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    s3.upload_file(feat_path, bucket, f"processed/features_v2/{patient_id}_features.npy")
    s3.upload_file(label_path, bucket, f"processed/features_v2/{patient_id}_labels.npy")
    print(f"Uploaded to s3://{bucket}/processed/features_v2/")

    # Clean up raw windows to keep disk usage flat
    os.remove(x_path)
    os.remove(y_path)
    print(f"Cleaned up local {patient_id}_X.npy, {patient_id}_y.npy")


if __name__ == "__main__":
    patient_id = sys.argv[1]
    process_patient(patient_id)