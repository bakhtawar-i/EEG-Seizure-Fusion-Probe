"""
Run inference with the saved LaBraM checkpoint on the test set, saving
per-window predictions tagged by patient and original pickle filename
(needed to map back to seizure timing).
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import boto3
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "/home/ubuntu/LaBraM")
sys.path.insert(0, "/home/ubuntu/EEG-Seizure-Fusion-Probe/scripts")

import modeling_finetune
import utils

TEST_PATIENTS = ["chb17", "chb18", "chb19", "chb20"]
DATA_ROOT = "/home/ubuntu/EEG-Seizure-Fusion-Probe/data/processed/labram"
CH_NAMES = ["FP1", "F7", "T7", "P7", "FP1", "F3", "C3", "P3",
            "FP2", "F4", "C4", "P4", "FP2", "F8", "T8", "P8"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = modeling_finetune.__dict__["labram_base_patch200_200"](
    pretrained=False,
    num_classes=1,
    drop_rate=0.0,
    drop_path_rate=0.1,
    attn_drop_rate=0.0,
    use_mean_pooling=True,
    init_scale=0.001,
    use_rel_pos_bias=True,
    use_abs_pos_emb=True,
    init_values=0.1,
    qkv_bias=True,
)
checkpoint = torch.load(
    "/home/ubuntu/LaBraM/checkpoints_chbmit_10ep_final/checkpoint-4.pth",
    map_location="cpu", weights_only=False
)
model.load_state_dict(checkpoint["model"], strict=False)
model.to(device)
model.eval()

input_chans = utils.get_input_chans(CH_NAMES)

rows = []
for p in TEST_PATIENTS:
    files = sorted(glob.glob(os.path.join(DATA_ROOT, p, "*.pkl")))
    for fpath in files:
        with open(fpath, "rb") as f:
            sample = pickle.load(f)
        X = torch.FloatTensor(sample["X"]).unsqueeze(0).to(device) / 100
        X = X.view(1, 16, 10, 200)  # (B, N, A, T) matching their rearrange convention
        with torch.no_grad():
            logits = model(X, input_chans=input_chans)
            prob = torch.sigmoid(logits).item()
        rows.append({
            "patient": p,
            "filename": os.path.basename(fpath),
            "y_true": sample["y"],
            "labram_prob": prob,
        })

df = pd.DataFrame(rows)
os.makedirs("/home/ubuntu/EEG-Seizure-Fusion-Probe/saved_models", exist_ok=True)
out_path = "/home/ubuntu/EEG-Seizure-Fusion-Probe/saved_models/labram_test_predictions.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} predictions to {out_path}")

s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
bucket = os.environ["S3_BUCKET_NAME"]
s3.upload_file(out_path, bucket, "models/labram/labram_test_predictions.csv")
print("Uploaded to S3")