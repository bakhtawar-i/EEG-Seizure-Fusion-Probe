"""
Preprocess one CHB-MIT patient: filter, window, label, save + upload to S3.

Usage:
    uv run python scripts/02_preprocess_patient.py chb01
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import mne
import boto3
from dotenv import load_dotenv

load_dotenv()

WINDOW_SEC = 4
SFREQ = 256
WINDOW_SAMPLES = WINDOW_SEC * SFREQ

# Canonical 22 unique channels from chb01's header, duplicate T8-P8 removed.
# NOTE: verify this list matches chb02/chb03 exactly before trusting it blindly —
# CHB-MIT is known to vary channel sets across a few patients.
CANONICAL_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
]


# def load_and_filter(edf_path: str) -> mne.io.Raw:
#     raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

#     # MNE appends -0/-1 to duplicate channel names (e.g. T8-P8 -> T8-P8-0, T8-P8-1).
#     # Strip suffixes and drop the resulting duplicate, keeping the first occurrence.
#     rename_map = {}
#     for ch in raw.ch_names:
#         stripped = ch.rsplit("-0", 1)[0] if ch.endswith("-0") else ch
#         stripped = stripped.rsplit("-1", 1)[0] if ch.endswith("-1") else stripped
#         rename_map[ch] = stripped
#     raw.rename_channels(rename_map, allow_duplicate_names=True)

#     # Drop any now-duplicate channels, keep first
#     seen = set()
#     drop = []
#     for ch in raw.ch_names:
#         if ch in seen:
#             drop.append(ch)
#         seen.add(ch)
#     if drop:
#         raw.drop_channels(drop)

#     missing = [c for c in CANONICAL_CHANNELS if c not in raw.ch_names]
#     if missing:
#         raise ValueError(f"{edf_path} missing expected channels: {missing}")

#     raw.pick(CANONICAL_CHANNELS)  # also reorders to match canonical order
#     raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
#     return raw

def load_and_filter(edf_path: str) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # MNE appends -0/-1 to duplicate channel names (e.g. T8-P8 -> T8-P8-0, T8-P8-1).
    # Drop the second occurrence BEFORE renaming, so we never hit a duplicate-name
    # collision (avoids needing allow_duplicate_names, which isn't in all MNE versions).
    seen_stripped = set()
    drop = []
    rename_map = {}
    for ch in raw.ch_names:
        stripped = ch
        if stripped.endswith("-0") or stripped.endswith("-1"):
            stripped = stripped.rsplit("-", 1)[0]
        if stripped in seen_stripped:
            drop.append(ch)
        else:
            seen_stripped.add(stripped)
            rename_map[ch] = stripped

    if drop:
        raw.drop_channels(drop)
    raw.rename_channels(rename_map)

    missing = [c for c in CANONICAL_CHANNELS if c not in raw.ch_names]
    if missing:
        raise ValueError(f"{edf_path} missing expected channels: {missing}")

    raw.pick(CANONICAL_CHANNELS)
    raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    return raw


def window_and_label(raw: mne.io.Raw, seizure_rows: pd.DataFrame):
    data = raw.get_data()  # shape: (n_channels, n_samples)
    n_samples = data.shape[1]
    n_windows = n_samples // WINDOW_SAMPLES

    windows = []
    labels = []
    for i in range(n_windows):
        start_sample = i * WINDOW_SAMPLES
        end_sample = start_sample + WINDOW_SAMPLES
        start_sec = start_sample / SFREQ
        end_sec = end_sample / SFREQ

        label = 0
        for _, row in seizure_rows.iterrows():
            if pd.isna(row["seizure_start_sec"]):
                continue
            # window overlaps a labeled seizure interval
            if start_sec < row["seizure_end_sec"] and end_sec > row["seizure_start_sec"]:
                label = 1
                break

        windows.append(data[:, start_sample:end_sample])
        labels.append(label)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def process_patient(patient_id: str):
    edf_dir = f"data/raw/chbmit/{patient_id}"
    labels_csv = f"data/processed/labels/{patient_id}_labels.csv"
    label_df = pd.read_csv(labels_csv)

    all_windows = []
    all_labels = []

    edf_files = sorted(glob.glob(os.path.join(edf_dir, f"{patient_id}_*.edf")))
    for edf_path in edf_files:
        fname = os.path.basename(edf_path)
        file_rows = label_df[label_df["filename"] == fname]

        if file_rows.empty:
            print(f"[SKIP] {fname}: not found in labels CSV")
            continue

        raw = load_and_filter(edf_path)
        windows, labels = window_and_label(raw, file_rows)
        all_windows.append(windows)
        all_labels.append(labels)
        print(f"{fname}: {len(labels)} windows, {labels.sum()} seizure windows")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\n{patient_id} totals: {X.shape[0]} windows, {y.sum()} seizure ({y.mean()*100:.2f}%)")

    out_dir = "data/processed/windows"
    os.makedirs(out_dir, exist_ok=True)
    x_path = os.path.join(out_dir, f"{patient_id}_X.npy")
    y_path = os.path.join(out_dir, f"{patient_id}_y.npy")
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Saved locally to {x_path}, {y_path}")

    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    s3.upload_file(x_path, bucket, f"processed/windows/{patient_id}_X.npy")
    s3.upload_file(y_path, bucket, f"processed/windows/{patient_id}_y.npy")
    print(f"Uploaded to s3://{bucket}/processed/windows/{patient_id}_X.npy (+ _y.npy)")


if __name__ == "__main__":
    patient_id = sys.argv[1]
    process_patient(patient_id)