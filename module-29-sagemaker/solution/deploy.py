"""
Deploy a trained FinBERT model and configure auto-scaling.

Run after the pipeline completes:
    python deploy.py --model-s3 <uri> --role <ARN> 
"""

import argparse
import json

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.serve.model_builder import ModelBuilder

ENDPOINT_NAME = "finbert-sentiment-endpoint"


# TODO: Deploy the trained model to a SageMaker managed endpoint
# https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-modelbuilder-creation.html#how-it-works-modelbuilder-creation-deploy
def deploy_endpoint(role: str, model_s3: str, session: Session):
    region = session.boto_region_name

    inference_image = image_uris.retrieve(
        framework="huggingface",
        region=region,
        version="4.26.0",
        py_version="py39",
        base_framework_version="pytorch1.13.1",
        image_scope="inference",
        instance_type="ml.m5.large",
    )

    #TODO Create a ModelBuilder that points to the trained model artifact in S3 and uses the HuggingFace inference image.
    model_builder = ModelBuilder(
        s3_model_data_url=model_s3,
        image_uri=inference_image,
        role_arn=role,
        sagemaker_session=session,
    )

    #TODO Build the model
    print(f"Building model from: {model_s3}")
    model_builder.build()

    #TODO deploy the model
    print(f"Deploying to endpoint: {ENDPOINT_NAME} ...")
    predictor = model_builder.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"Endpoint ready: {predictor.endpoint_name}")
    return predictor


# TODO: Configure a target-tracking auto-scaling policy on the endpoint.
#   - Metric  : SageMakerVariantInvocationsPerInstance: requests per instance per minute
#   - Target  : 1000 invocations/instance/min
#   - Min/Max : 1-4 instances
#   - ScaleOut cooldown : 60 s
#   - ScaleIn  cooldown : 300 s
def configure_autoscaling(endpoint_name: str, region: str):
    # https://docs.aws.amazon.com/boto3/latest/reference/services/application-autoscaling.html
    aas = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    #TODO: Register the endpoint as a scalable target with the specified min and max capacity.
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=4,
    )

    #TODO: Create a target-tracking scaling policy that maintains an average of 1000 invocations per instance per minute.
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
    print("Auto-scaling configured: 1-4 instances, target 1000 invocations/instance")


# TODO: Send a test request to the endpoint to verify it returns predictions.
# return the sentiment and confidence score
# https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-test-endpoints.html
def verify_endpoint(endpoint_name: str, region: str):
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    test_inputs = [
        "The company reported record profits and strong revenue growth.",
        "The stock plummeted after missing earnings expectations.",
        "Trading volume remained steady with no major changes.",
    ]

    print("\nVerifying endpoint predictions:")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"inputs": test_inputs}),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    for text, pred in zip(test_inputs, result):
        label = id2label.get(int(pred["label"]), str(pred["label"]))
        print(f"  [Sentiment: {label:8} - Confidence: {pred['score']:.2f}]  {text[:60]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3", required=True,
                        help="S3 URI of model.tar.gz (from training job output)")
    parser.add_argument("--role", default=None)
    args = parser.parse_args()

    session = Session()
    role    = args.role or get_execution_role()
    region  = session.boto_region_name

    predictor = deploy_endpoint(role, args.model_s3, session)
    configure_autoscaling(predictor.endpoint_name, region)
    
    verify_endpoint(predictor.endpoint_name, region)

    print(f"\nDeployment complete.")
    print(f"Endpoint: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
