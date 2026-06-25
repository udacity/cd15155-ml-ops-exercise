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
    import time
    sm = boto3.client("sagemaker", region_name=region)

    try:
        paginator = sm.get_paginator("list_model_packages")
        for page in paginator.paginate(ModelPackageGroupName=MODEL_PACKAGE_GROUP):
            for pkg in page["ModelPackageSummaryList"]:
                sm.delete_model_package(ModelPackageName=pkg["ModelPackageArn"])
                print(f"Model package deleted: {pkg['ModelPackageArn']}")

        time.sleep(10)
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


def delete_domain(region):
    sm = boto3.client("sagemaker", region_name=region)
    try:
        domains = sm.list_domains()["Domains"]
        if not domains:
            print("No SageMaker domains found.")
            return

        for domain in domains:
            domain_id = domain["DomainId"]
            print(f"Cleaning up domain: {domain_id}...")

            # Delete all apps in all user profiles
            profiles = sm.list_user_profiles(DomainIdEquals=domain_id)["UserProfiles"]
            for profile in profiles:
                username = profile["UserProfileName"]
                apps = sm.list_apps(
                    DomainIdEquals=domain_id,
                    UserProfileNameEquals=username,
                )["Apps"]
                for app in apps:
                    if app["Status"] not in ("Deleted", "Deleting"):
                        sm.delete_app(
                            DomainId=domain_id,
                            UserProfileName=username,
                            AppType=app["AppType"],
                            AppName=app["AppName"],
                        )
                        print(f"  App deleted: {app['AppName']}")

            # Delete all user profiles
            for profile in profiles:
                sm.delete_user_profile(
                    DomainId=domain_id,
                    UserProfileName=profile["UserProfileName"],
                )
                print(f"  User profile deleted: {profile['UserProfileName']}")

            # Delete the domain
            sm.delete_domain(DomainId=domain_id, RetentionPolicy={"HomeEfsFileSystem": "Delete"})
            print(f"Domain '{domain_id}' deleted.")
    except Exception as e:
        print(f"Domain deletion skipped: {e}")


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

    delete_endpoint(region)
    delete_model_registry(region)
    delete_pipeline(region)
    delete_s3_data(bucket, region)
    delete_domain(region)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
