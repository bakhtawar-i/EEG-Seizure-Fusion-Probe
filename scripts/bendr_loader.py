"""
CHB-MIT dataset loader for BENDR fine-tuning, using Phase 1's original
windowed data directly (18 channels, 512 samples @ 256Hz) — no separate
preprocessing needed, since BENDR's window length isn't fixed by the
pretrained checkpoint.

Auto-downloads from S3 if not present locally, matching the pattern used
throughout Phase 1/2/3.
"""
import os
import numpy as np
import torch
import boto3
from dotenv import load_dotenv

load_dotenv()

TRAIN_PATIENTS = [f"chb{str(i).zfill(2)}" for i in range(1, 17) if i != 12] + ["chb21", "chb22"]
TEST_PATIENTS = ["chb17", "chb18", "chb19", "chb20"]

DATA_DIR = "data/processed/windows"


def ensure_local(patient_id: str):
    x_path = f"{DATA_DIR}/{patient_id}_X.npy"
    y_path = f"{DATA_DIR}/{patient_id}_y.npy"
    if os.path.exists(x_path) and os.path.exists(y_path):
        return

    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(x_path):
        s3.download_file(bucket, f"processed/windows/{patient_id}_X.npy", x_path)
    if not os.path.exists(y_path):
        s3.download_file(bucket, f"processed/windows/{patient_id}_y.npy", y_path)


class CHBMITWindowDataset(torch.utils.data.Dataset):
    """
    Reads Phase 1's windowed .npy files per patient. Uses memory-mapped
    loading so multiple patients' data isn't fully resident in RAM at once.
    """
    def __init__(self, patients):
        self.index = []  # list of (patient_id, local_index_within_patient)
        self.arrays = {}  # patient_id -> (memmap X, in-memory y)

        for p in patients:
            ensure_local(p)
            X = np.load(f"{DATA_DIR}/{p}_X.npy", mmap_mode="r")
            y = np.load(f"{DATA_DIR}/{p}_y.npy")
            self.arrays[p] = (X, y)
            for i in range(len(y)):
                self.index.append((p, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        patient_id, local_idx = self.index[idx]
        X, y = self.arrays[patient_id]
        window = np.asarray(X[local_idx]).astype(np.float32)  # (18, 512)
        label = int(y[local_idx])
        return torch.from_numpy(window), label


def build_bendr_datasets():
    train_dataset = CHBMITWindowDataset(TRAIN_PATIENTS)
    test_dataset = CHBMITWindowDataset(TEST_PATIENTS)

    print(f"Train: {len(train_dataset)} windows across {len(TRAIN_PATIENTS)} patients")
    print(f"Test: {len(test_dataset)} windows across {len(TEST_PATIENTS)} patients")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_ds, test_ds = build_bendr_datasets()
    X, y = train_ds[0]
    print(f"Sample shape: {X.shape}, label: {y}")
    print(f"Sample dtype: {X.dtype}")