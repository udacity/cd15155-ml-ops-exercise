"""
Tear down all AWS resources created by pipeline.py.

Usage:
    python cleanup.py [--bucket <bucket>] [--region <region>]
"""

import argparse

import boto3
from sagemaker.core.helper.session_helper import Session

PIPELINE_NAME = "FinBERTPipeline"
S3_PREFIX = "finbert/"


def delete_pipeline(region):
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.delete_pipeline(PipelineName=PIPELINE_NAME)
        print(f"Pipeline '{PIPELINE_NAME}' deleted.")
    except Exception as e:
        print(f"Pipeline deletion skipped: {e}")


def delete_s3_data(bucket, region):
    s3 = boto3.resource("s3", region_name=region)
    try:
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=S3_PREFIX).delete()
        print(f"S3 data under s3://{bucket}/{S3_PREFIX} deleted.")
    except Exception as e:
        print(f"S3 cleanup skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    session = Session()
    region = args.region or session.boto_region_name
    bucket = args.bucket or session.default_bucket()

    print(f"Region : {region}")
    print(f"Bucket : {bucket}")
    print("Starting cleanup...")

    delete_pipeline(region)
    delete_s3_data(bucket, region)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
