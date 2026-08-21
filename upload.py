
import os, boto3
from dotenv import load_dotenv
load_dotenv()

s3 = boto3.client('s3', region_name=os.environ['AWS_DEFAULT_REGION'])
bucket = os.environ['S3_BUCKET_NAME']

local_dir = 'data/raw/chbmit'
for patient in sorted(os.listdir(local_dir)):
    patient_path = os.path.join(local_dir, patient)
    if not os.path.isdir(patient_path):
        continue
    if patient in ['chb01', 'chb02', 'chb03']:
        continue  # already processed, skip re-uploading raw
    for fname in os.listdir(patient_path):
        local_file = os.path.join(patient_path, fname)
        s3_key = f'raw/chbmit/{patient}/{fname}'
        print(f'Uploading {local_file} -> s3://{bucket}/{s3_key}')
        s3.upload_file(local_file, bucket, s3_key)
