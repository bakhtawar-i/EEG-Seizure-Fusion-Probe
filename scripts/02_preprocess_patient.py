"""
Preprocess one CHB-MIT patient: filter, window, label, save + upload to S3.
Memory-safe version — writes windows directly to a disk-backed array per file,
never holds the full patient's data in RAM at once.

Usage:
    uv run python scripts/02_preprocess_patient.py chb01
"""
import os
import sys
import glob
import gc
import numpy as np
import pandas as pd
import mne
import boto3
from dotenv import load_dotenv

load_dotenv()

WINDOW_SEC = 2
SFREQ = 256
WINDOW_SAMPLES = WINDOW_SEC * SFREQ
SEIZURE_STRIDE_SEC = 1
SEIZURE_MARGIN_SEC = 2
N_CHANNELS = 18

# CANONICAL_CHANNELS = [
#     "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
#     "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
#     "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8",
# ]

CANONICAL_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

def clean_channels(raw: mne.io.Raw) -> mne.io.Raw:
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
        raise ValueError(f"Missing expected channels: {missing}")

    raw.pick(CANONICAL_CHANNELS)
    return raw


def get_window_starts(n_samples: int, seizure_intervals: list) -> list:
    """Pure computation of window start sample indices — no raw data needed."""
    window_starts = set()

    n_full_windows = n_samples // WINDOW_SAMPLES
    for i in range(n_full_windows):
        window_starts.add(i * WINDOW_SAMPLES)

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

    return sorted(s for s in window_starts if s + WINDOW_SAMPLES <= n_samples)


def label_for(start_sec, end_sec, seizure_intervals):
    for s_start, s_end in seizure_intervals:
        if start_sec < s_end and end_sec > s_start:
            return 1
    return 0


def process_patient(patient_id: str):
    edf_dir = f"data/raw/chbmit/{patient_id}"
    labels_csv = f"data/processed/labels/{patient_id}_labels.csv"
    label_df = pd.read_csv(labels_csv)

    edf_files = sorted(glob.glob(os.path.join(edf_dir, f"{patient_id}_*.edf")))

    # --- Pass 1: count total windows across all files (cheap, preload=False) ---
    file_plans = []  # (edf_path, seizure_intervals, n_samples, window_starts)
    total_windows = 0
    for edf_path in edf_files:
        fname = os.path.basename(edf_path)
        file_rows = label_df[label_df["filename"] == fname]
        if file_rows.empty:
            print(f"[SKIP] {fname}: not found in labels CSV")
            continue

        raw_header = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        n_samples = raw_header.n_times
        seizure_intervals = [
            (row["seizure_start_sec"], row["seizure_end_sec"])
            for _, row in file_rows.iterrows()
            if not pd.isna(row["seizure_start_sec"])
        ]
        starts = get_window_starts(n_samples, seizure_intervals)
        file_plans.append((edf_path, seizure_intervals, starts))
        total_windows += len(starts)
        del raw_header

    print(f"{patient_id}: {total_windows} total windows planned across {len(file_plans)} files")

    # --- Allocate disk-backed array — never held fully in RAM ---
    out_dir = "data/processed/windows"
    os.makedirs(out_dir, exist_ok=True)
    x_path = os.path.join(out_dir, f"{patient_id}_X.npy")
    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float16, shape=(total_windows, N_CHANNELS, WINDOW_SAMPLES)
    )
    y = np.zeros(total_windows, dtype=np.int64)

    # --- Pass 2: process one file at a time, write directly into the memmap ---
    offset = 0
    for edf_path, seizure_intervals, starts in file_plans:
        fname = os.path.basename(edf_path)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw = clean_channels(raw)
        raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
        data = raw.get_data()  # (n_channels, n_samples), float64 — but only one file at a time

        n_seizure_this_file = 0
        for start_sample in starts:
            end_sample = start_sample + WINDOW_SAMPLES
            start_sec = start_sample / SFREQ
            end_sec = end_sample / SFREQ
            lbl = label_for(start_sec, end_sec, seizure_intervals)

            X[offset] = data[:, start_sample:end_sample].astype(np.float16)
            y[offset] = lbl
            if lbl == 1:
                n_seizure_this_file += 1
            offset += 1

        print(f"{fname}: {len(starts)} windows, {n_seizure_this_file} seizure windows")

        del raw, data
        gc.collect()  # explicit cleanup before moving to the next file

    X.flush()  # ensure everything is written to disk

    print(f"\n{patient_id} totals: {total_windows} windows, {y.sum()} seizure ({y.mean()*100:.2f}%)")

    y_path = os.path.join(out_dir, f"{patient_id}_y.npy")
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