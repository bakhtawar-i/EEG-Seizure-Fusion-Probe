"""
Parse a CHB-MIT chbXX-summary.txt file into a table of:
    filename, seizure_start_sec, seizure_end_sec

A file with no seizures gets no rows. A file with multiple seizures
gets one row per seizure (some CHB-MIT files have >1 seizure).

Usage:
    uv run python scripts/01_parse_summary.py data/raw/chbmit/chb01/chb01-summary.txt
"""
import os
import re
import sys
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def parse_summary(path: str) -> pd.DataFrame:
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    blocks = re.split(r"(?=File Name: )", text)

    rows = []
    for block in blocks:
        fname_match = re.search(r"File Name:\s*(\S+)", block)
        if not fname_match:
            continue
        fname = fname_match.group(1)

        n_seiz_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        n_seizures = int(n_seiz_match.group(1)) if n_seiz_match else 0

        if n_seizures == 0:
            rows.append({"filename": fname, "seizure_start_sec": None, "seizure_end_sec": None})
            continue

        starts = re.findall(r"Seizure(?:\s*\d*)?\s*Start Time:\s*(\d+)\s*seconds", block)
        ends = re.findall(r"Seizure(?:\s*\d*)?\s*End Time:\s*(\d+)\s*seconds", block)

        if len(starts) != n_seizures or len(ends) != n_seizures:
            print(f"[WARNING] {fname}: expected {n_seizures} seizures, "
                  f"found {len(starts)} starts / {len(ends)} ends. Check file format manually.")

        for s, e in zip(starts, ends):
            rows.append({"filename": fname, "seizure_start_sec": int(s), "seizure_end_sec": int(e)})

    return pd.DataFrame(rows)


def upload_to_s3(local_path: str, s3_key: str):
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["S3_BUCKET_NAME"]
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded to s3://{bucket}/{s3_key}")


if __name__ == "__main__":
    summary_path = sys.argv[1]
    df = parse_summary(summary_path)
    print(df.to_string(index=False))
    print(f"\nTotal rows: {len(df)}")
    print(f"Files with seizures: {df['seizure_start_sec'].notna().sum()}")

    base = os.path.basename(summary_path)
    patient_id = base.split("-summary")[0]

    out_dir = "data/processed/labels"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{patient_id}_labels.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved locally to {out_path}")

    upload_to_s3(out_path, f"processed/labels/{patient_id}_labels.csv")


# *** OUTPUT ***

# this is for patient 1, run it for patient 2 and 3

#     filename  seizure_start_sec  seizure_end_sec
# chb01_01.edf                NaN              NaN
# chb01_02.edf                NaN              NaN
# chb01_03.edf             2996.0           3036.0
# chb01_04.edf             1467.0           1494.0
# chb01_05.edf                NaN              NaN
# chb01_06.edf                NaN              NaN
# chb01_07.edf                NaN              NaN
# chb01_08.edf                NaN              NaN
# chb01_09.edf                NaN              NaN
# chb01_10.edf                NaN              NaN
# chb01_11.edf                NaN              NaN
# chb01_12.edf                NaN              NaN
# chb01_13.edf                NaN              NaN
# chb01_14.edf                NaN              NaN
# chb01_15.edf             1732.0           1772.0
# chb01_16.edf             1015.0           1066.0
# chb01_17.edf                NaN              NaN
# chb01_18.edf             1720.0           1810.0
# chb01_19.edf                NaN              NaN
# chb01_20.edf                NaN              NaN
# chb01_21.edf              327.0            420.0
# chb01_22.edf                NaN              NaN
# chb01_23.edf                NaN              NaN
# chb01_24.edf                NaN              NaN
# chb01_25.edf                NaN              NaN
# chb01_26.edf             1862.0           1963.0
# chb01_27.edf                NaN              NaN
# chb01_29.edf                NaN              NaN
# chb01_30.edf                NaN              NaN
# chb01_31.edf                NaN              NaN
# chb01_32.edf                NaN              NaN
# chb01_33.edf                NaN              NaN
# chb01_34.edf                NaN              NaN
# chb01_36.edf                NaN              NaN
# chb01_37.edf                NaN              NaN
# chb01_38.edf                NaN              NaN
# chb01_39.edf                NaN              NaN
# chb01_40.edf                NaN              NaN
# chb01_41.edf                NaN              NaN
# chb01_42.edf                NaN              NaN
# chb01_43.edf                NaN              NaN
# chb01_46.edf                NaN              NaN

# Total rows: 42
# Files with seizures: 7
