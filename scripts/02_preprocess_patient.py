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

# WINDOW_SEC = 4
# SFREQ = 256
# WINDOW_SAMPLES = WINDOW_SEC * SFREQ

# Canonical 22 unique channels from chb01's header, duplicate T8-P8 removed.
# NOTE: verify this list matches chb02/chb03 exactly before trusting it blindly —
# CHB-MIT is known to vary channel sets across a few patients.
CANONICAL_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
]

WINDOW_SEC = 2
SFREQ = 256
WINDOW_SAMPLES = WINDOW_SEC * SFREQ
SEIZURE_STRIDE_SEC = 1       # overlapping stride within seizure regions
SEIZURE_MARGIN_SEC = 2       # extra context around each seizure interval


def window_and_label(raw: mne.io.Raw, seizure_rows: pd.DataFrame):
    data = raw.get_data()  # shape: (n_channels, n_samples)
    n_samples = data.shape[1]

    seizure_intervals = [
        (row["seizure_start_sec"], row["seizure_end_sec"])
        for _, row in seizure_rows.iterrows()
        if not pd.isna(row["seizure_start_sec"])
    ]

    def label_for(start_sec, end_sec):
        for s_start, s_end in seizure_intervals:
            if start_sec < s_end and end_sec > s_start:
                return 1
        return 0

    window_starts = set()

    # Pass 1: non-overlapping sweep across the full recording (negatives + sparse positives)
    n_full_windows = n_samples // WINDOW_SAMPLES
    for i in range(n_full_windows):
        start_sample = i * WINDOW_SAMPLES
        window_starts.add(start_sample)

    # Pass 2: overlapping sweep restricted to seizure intervals (+ margin), 1s stride
    stride_samples = SEIZURE_STRIDE_SEC * SFREQ
    for s_start, s_end in seizure_intervals:
        region_start_sec = max(0, s_start - SEIZURE_MARGIN_SEC)
        region_end_sec = min(n_samples / SFREQ, s_end + SEIZURE_MARGIN_SEC)
        region_start_sample = int(region_start_sec * SFREQ)
        region_end_sample = int(region_end_sec * SFREQ) - WINDOW_SAMPLES

        start = region_start_sample
        while start <= region_end_sample:
            window_starts.add(start)
            start += stride_samples

    windows = []
    labels = []
    for start_sample in sorted(window_starts):
        end_sample = start_sample + WINDOW_SAMPLES
        if end_sample > n_samples:
            continue
        start_sec = start_sample / SFREQ
        end_sec = end_sample / SFREQ
        label = label_for(start_sec, end_sec)

        windows.append(data[:, start_sample:end_sample])
        labels.append(label)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)

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


# def window_and_label(raw: mne.io.Raw, seizure_rows: pd.DataFrame):
#     data = raw.get_data()  # shape: (n_channels, n_samples)
#     n_samples = data.shape[1]
#     n_windows = n_samples // WINDOW_SAMPLES

#     windows = []
#     labels = []
#     for i in range(n_windows):
#         start_sample = i * WINDOW_SAMPLES
#         end_sample = start_sample + WINDOW_SAMPLES
#         start_sec = start_sample / SFREQ
#         end_sec = end_sample / SFREQ

#         label = 0
#         for _, row in seizure_rows.iterrows():
#             if pd.isna(row["seizure_start_sec"]):
#                 continue
#             # window overlaps a labeled seizure interval
#             if start_sec < row["seizure_end_sec"] and end_sec > row["seizure_start_sec"]:
#                 label = 1
#                 break

#         windows.append(data[:, start_sample:end_sample])
#         labels.append(label)

#     return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


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

    X = X.astype(np.float16)  # halves file size

    print(f"\n{patient_id} totals: {X.shape[0]} windows, {y.sum()} seizure ({y.mean()*100:.2f}%)")

    out_dir = "data/processed/windows"
    os.makedirs(out_dir, exist_ok=True)
    x_path = os.path.join(out_dir, f"{patient_id}_X.npz")
    np.savez_compressed(x_path, X=X, y=y)
    print(f"Saved locally to {x_path}")

    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    s3.upload_file(x_path, bucket, f"processed/windows/{patient_id}.npz")
    print(f"Uploaded to s3://{bucket}/processed/windows/{patient_id}.npz")


if __name__ == "__main__":
    patient_id = sys.argv[1]
    process_patient(patient_id)