"""
Preprocess CHB-MIT raw EDFs to match LaBraM's expected input format.
Separate from Phase 1/2 pipeline — different filter band, sample rate,
window length, and channel set (16 vs. 18, bipolar-compatible only).

Output: one .pkl file per 10-second window: {"X": array(16, 2000), "y": 0/1}

Usage:
    uv run python scripts/08_preprocess_for_labram.py chb04
"""
import os
import sys
import glob
import gc
import pickle
import numpy as np
import pandas as pd
import mne
import boto3
from dotenv import load_dotenv

load_dotenv()

TARGET_SFREQ = 200
WINDOW_SAMPLES = 2000  # 10 seconds @ 200Hz, matches LaBraM's TUAB convention
ORIG_SFREQ = 256

# 16 channels confirmed compatible with LaBraM's standard_1020 list
# (FZ-CZ, CZ-PZ excluded — not in LaBraM's channel vocabulary)
LABRAM_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
]


def ensure_raw_local(patient_id: str):
    """Raw EDFs may need re-downloading if not present (Phase 1 raw data
    was cleared from disk after processing)."""
    edf_dir = f"data/raw/chbmit/{patient_id}"
    if os.path.isdir(edf_dir) and glob.glob(os.path.join(edf_dir, "*.edf")):
        return
    print(f"  Raw EDFs for {patient_id} not found locally — downloading from PhysioNet...")
    os.makedirs("data/raw/chbmit", exist_ok=True)
    os.system(
        f"cd data/raw/chbmit && wget -q -r -N -c -np -nH --cut-dirs=3 "
        f"https://physionet.org/files/chbmit/1.0.0/{patient_id}/"
    )


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

    missing = [c for c in LABRAM_CHANNELS if c not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing expected channels: {missing}")

    raw.pick(LABRAM_CHANNELS)  # also reorders to match canonical order
    return raw


def get_window_starts(n_samples_target: int) -> list:
    """Non-overlapping windows across the full (resampled) recording."""
    n_windows = n_samples_target // WINDOW_SAMPLES
    return [i * WINDOW_SAMPLES for i in range(n_windows)]


def label_for(start_sec, end_sec, seizure_intervals):
    for s_start, s_end in seizure_intervals:
        if start_sec < s_end and end_sec > s_start:
            return 1
    return 0


def process_patient(patient_id: str):
    ensure_raw_local(patient_id)

    edf_dir = f"data/raw/chbmit/{patient_id}"
    labels_csv = f"data/processed/labels/{patient_id}_labels.csv"
    label_df = pd.read_csv(labels_csv)

    out_dir = f"data/processed/labram/{patient_id}"
    os.makedirs(out_dir, exist_ok=True)

    edf_files = sorted(glob.glob(os.path.join(edf_dir, f"{patient_id}_*.edf")))
    total_windows = 0
    total_seizure = 0

    for edf_path in edf_files:
        fname = os.path.basename(edf_path)
        file_rows = label_df[label_df["filename"] == fname]
        if file_rows.empty:
            print(f"  [SKIP] {fname}: not found in labels CSV")
            continue

        seizure_intervals = [
            (row["seizure_start_sec"], row["seizure_end_sec"])
            for _, row in file_rows.iterrows()
            if not pd.isna(row["seizure_start_sec"])
        ]

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw = clean_channels(raw)

        # LaBraM's exact preprocessing spec
        raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
        raw.notch_filter(50.0, verbose=False)
        raw.resample(TARGET_SFREQ, n_jobs=1, verbose=False)

        data = raw.get_data(units="uV")  # (16, n_samples_at_200hz)
        n_samples = data.shape[1]

        starts = get_window_starts(n_samples)
        n_seizure_this_file = 0

        for start_sample in starts:
            end_sample = start_sample + WINDOW_SAMPLES
            start_sec = start_sample / TARGET_SFREQ
            end_sec = end_sample / TARGET_SFREQ
            lbl = label_for(start_sec, end_sec, seizure_intervals)

            window = data[:, start_sample:end_sample].astype(np.float32)
            dump_path = os.path.join(
                out_dir, f"{fname.split('.')[0]}_{start_sample}.pkl"
            )
            with open(dump_path, "wb") as f:
                pickle.dump({"X": window, "y": lbl}, f)

            total_windows += 1
            if lbl == 1:
                n_seizure_this_file += 1
                total_seizure += 1

        print(f"  {fname}: {len(starts)} windows, {n_seizure_this_file} seizure")

        del raw, data
        gc.collect()

    print(f"\n{patient_id} totals: {total_windows} windows, {total_seizure} seizure "
          f"({100*total_seizure/total_windows:.2f}%)")

    # Upload the whole patient folder to S3
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    for pkl_file in os.listdir(out_dir):
        local_path = os.path.join(out_dir, pkl_file)
        s3_key = f"processed/labram/{patient_id}/{pkl_file}"
        s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded {len(os.listdir(out_dir))} files to s3://{bucket}/processed/labram/{patient_id}/")

    # Clean up local raw EDFs to keep disk usage flat
    import shutil
    shutil.rmtree(edf_dir, ignore_errors=True)
    print(f"Cleaned up local raw EDFs for {patient_id}")


if __name__ == "__main__":
    patient_id = sys.argv[1]
    process_patient(patient_id)