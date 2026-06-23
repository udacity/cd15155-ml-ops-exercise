"""
Tear down all AWS resources created by pipeline.py and deploy.py.

Usage:
    python cleanup.py [--bucket <bucket>] [--region <region>]
"""

import argparse

import boto3
from sagemaker.core.helper.session_helper import Session

PIPELINE_NAME = "FinBERTPipeline"
MODEL_PACKAGE_GROUP = "FinBERTSentimentClassifiers"
ENDPOINT_NAME = "finbert-sentiment-endpoint"
MODEL_NAME = "finbert-from-registry"
S3_PREFIX = "finbert/"


def delete_endpoint(region):
    sm = boto3.client("sagemaker", region_name=region)

    # Remove auto-scaling first
    aas = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{ENDPOINT_NAME}/variant/AllTraffic"
    try:
        aas.delete_scaling_policy(
            PolicyName=f"{ENDPOINT_NAME}-scaling",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
        aas.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
        print("Auto-scaling removed.")
    except Exception as e:
        print(f"Auto-scaling removal skipped: {e}")

    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint '{ENDPOINT_NAME}' deleted.")
    except Exception as e:
        print(f"Endpoint deletion skipped: {e}")

    try:
        sm.delete_model(ModelName=MODEL_NAME)
        print(f"Model '{MODEL_NAME}' deleted.")
    except Exception as e:
        print(f"Model deletion skipped: {e}")


def delete_model_registry(region):
    sm = boto3.client("sagemaker", region_name=region)

    # Delete all model packages in the group
    try:
        paginator = sm.get_paginator("list_model_packages")
        for page in paginator.paginate(ModelPackageGroupName=MODEL_PACKAGE_GROUP):
            for pkg in page["ModelPackageSummaryList"]:
                sm.delete_model_package(ModelPackageName=pkg["ModelPackageArn"])
                print(f"Model package deleted: {pkg['ModelPackageArn']}")

        sm.delete_model_package_group(ModelPackageGroupName=MODEL_PACKAGE_GROUP)
        print(f"Model package group '{MODEL_PACKAGE_GROUP}' deleted.")
    except Exception as e:
        print(f"Model registry cleanup skipped: {e}")


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
        deleted = bucket_obj.objects.filter(Prefix=S3_PREFIX).delete()
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

    delete_endpoint(region)
    delete_model_registry(region)
    delete_pipeline(region)
    delete_s3_data(bucket, region)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
