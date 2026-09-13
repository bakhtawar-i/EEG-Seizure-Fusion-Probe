"""
Run inference with the saved BENDR checkpoint on the test set, saving
per-window predictions tagged by patient and window index (to map back
to seizure timing via the same window-index convention as GBM/Phase 1).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import boto3
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, "scripts")

from bendr_loader import build_bendr_datasets, TEST_PATIENTS, DATA_DIR
from bendr_utils import load_pretrained_interpolated_bendr
from chbmit_chs_info import get_chbmit_chs_info

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chs_info = get_chbmit_chs_info()
model = load_pretrained_interpolated_bendr(chs_info, n_outputs=2)
model.load_state_dict(torch.load("bendr_checkpoints_frozen/checkpoint-9.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

rows = []
for p in TEST_PATIENTS:
    X = np.load(f"{DATA_DIR}/{p}_X.npy", mmap_mode="r")
    y = np.load(f"{DATA_DIR}/{p}_y.npy")

    for i in range(len(y)):
        window = torch.from_numpy(np.asarray(X[i]).astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(window)[:, 1]
            prob = torch.sigmoid(logits).float().item()
        rows.append({
            "patient": p,
            "window_index": i,
            "y_true": int(y[i]),
            "bendr_prob": prob,
        })

df = pd.DataFrame(rows)
os.makedirs("saved_models", exist_ok=True)
out_path = "saved_models/bendr_test_predictions.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} predictions to {out_path}")

s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
bucket = os.environ["S3_BUCKET_NAME"]
s3.upload_file(out_path, bucket, "models/bendr/bendr_test_predictions.csv")
print("Uploaded to S3")