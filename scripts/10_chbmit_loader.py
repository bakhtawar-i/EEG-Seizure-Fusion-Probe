"""
CHB-MIT dataset loader for LaBraM fine-tuning, modeled on their TUABLoader.
Reads pickle files produced by scripts/08_preprocess_for_labram.py.
"""
import os
import glob
import pickle
import torch
from scipy.signal import resample


TRAIN_PATIENTS = [f"chb{str(i).zfill(2)}" for i in range(1, 17)] + ["chb21", "chb22"]
TEST_PATIENTS = ["chb17", "chb18", "chb19", "chb20"]

DATA_ROOT = "data/processed/labram"


def get_patient_files(patients):
    """Returns list of (full_path) for every .pkl window belonging to the given patients."""
    files = []
    for p in patients:
        patient_dir = os.path.join(DATA_ROOT, p)
        files.extend(sorted(glob.glob(os.path.join(patient_dir, "*.pkl"))))
    return files


class CHBMITLoader(torch.utils.data.Dataset):
    """
    Matches LaBraM's TUABLoader interface: __getitem__ returns (X, y) as
    torch tensors. X shape: (16 channels, 2000 samples) @ 200Hz, already
    matches LaBraM's expected sampling rate (no on-the-fly resample needed
    unless sampling_rate arg differs from 200).
    """
    def __init__(self, files, sampling_rate=200):
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], "rb") as f:
            sample = pickle.load(f)
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(10 * self.sampling_rate), axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


def build_chbmit_datasets(sampling_rate=200):
    train_files = get_patient_files(TRAIN_PATIENTS)
    test_files = get_patient_files(TEST_PATIENTS)

    print(f"Train: {len(train_files)} windows across {len(TRAIN_PATIENTS)} patients")
    print(f"Test: {len(test_files)} windows across {len(TEST_PATIENTS)} patients")

    train_dataset = CHBMITLoader(train_files, sampling_rate=sampling_rate)
    test_dataset = CHBMITLoader(test_files, sampling_rate=sampling_rate)

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Quick sanity check
    train_ds, test_ds = build_chbmit_datasets()
    X, y = train_ds[0]
    print(f"Sample shape: {X.shape}, label: {y}")
    print(f"Sample dtype: {X.dtype}")