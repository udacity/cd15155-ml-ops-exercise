"""
Solution: Deploy an approved model from the Model Registry and configure auto-scaling.

Steps:
  1. Fetch the latest approved model package from the Model Registry
  2. Deploy it to a SageMaker managed endpoint
  3. Verify the endpoint returns predictions
  4. Configure auto-scaling: 1–4 instances, target 1000 invocations/instance

Run after approving the model in SageMaker Model Registry:
    python deploy.py
    python deploy.py --role <ARN>  (outside Studio)
"""

import argparse
import json

import boto3
import sagemaker
from sagemaker import ModelPackage

ENDPOINT_NAME        = "finbert-sentiment-endpoint"
MODEL_PACKAGE_GROUP  = "FinBERTSentimentClassifiers"


# TODO: Retrieve the latest approved model package ARN from the Model Registry
def get_latest_approved_model(sm_client, group_name: str) -> str:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    packages = response["ModelPackageSummaryList"]
    if not packages:
        raise ValueError(f"No approved models in group '{group_name}'. Approve one first.")
    arn = packages[0]["ModelPackageArn"]
    print(f"Using model package: {arn}")
    return arn


# TODO: Deploy the approved model to a SageMaker managed endpoint
def deploy_endpoint(role: str, session: sagemaker.Session) -> sagemaker.Predictor:
    sm_client = session.sagemaker_client
    model_arn  = get_latest_approved_model(sm_client, MODEL_PACKAGE_GROUP)

    model = ModelPackage(
        role=role,
        model_package_arn=model_arn,
        sagemaker_session=session,
    )

    print(f"Deploying to endpoint: {ENDPOINT_NAME} ...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"Endpoint ready: {predictor.endpoint_name}")
    return predictor


# TODO: Configure a target-tracking auto-scaling policy
def configure_autoscaling(endpoint_name: str, region: str):
    aas = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=4,
    )

    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scaling",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 1000.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    print(f"Auto-scaling configured: 1–4 instances, target 1000 invocations/instance")


def verify_endpoint(predictor: sagemaker.Predictor):
    test_inputs = [
        "The company reported record profits and strong revenue growth.",
        "The stock plummeted after missing earnings expectations.",
        "Trading volume remained steady with no major changes.",
    ]

    print("\nVerifying endpoint predictions:")
    predictor.serializer   = sagemaker.serializers.JSONSerializer()
    predictor.deserializer = sagemaker.deserializers.JSONDeserializer()

    result = predictor.predict({"inputs": test_inputs})
    for text, pred in zip(test_inputs, result):
        print(f"  [{pred['label']:8s} {pred['score']:.2f}]  {text[:60]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=None)
    args = parser.parse_args()

    session = sagemaker.Session(boto_session=boto3.Session())
    role    = args.role or sagemaker.get_execution_role()
    region  = session.boto_region_name

    predictor = deploy_endpoint(role, session)
    configure_autoscaling(predictor.endpoint_name, region)
    verify_endpoint(predictor)

    print(f"\nDeployment complete.")
    print(f"Endpoint: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
