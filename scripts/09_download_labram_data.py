"""
Download all LaBraM-format pickle files from S3 to local disk for training.
"""
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
bucket = os.environ["S3_BUCKET_NAME"]

local_root = "data/processed/labram"
os.makedirs(local_root, exist_ok=True)

paginator = s3.get_paginator("list_objects_v2")
total_downloaded = 0

for page in paginator.paginate(Bucket=bucket, Prefix="processed/labram/"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        # key format: processed/labram/chbXX/filename.pkl
        rel_path = key.replace("processed/labram/", "")
        local_path = os.path.join(local_root, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if not os.path.exists(local_path):
            s3.download_file(bucket, key, local_path)
            total_downloaded += 1

        if total_downloaded % 5000 == 0 and total_downloaded > 0:
            print(f"  {total_downloaded} files downloaded so far...")

print(f"\nDone. {total_downloaded} files downloaded to {local_root}/")