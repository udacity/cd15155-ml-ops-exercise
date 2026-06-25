"""
Deploy a trained FinBERT model from the SageMaker Model Registry.

Approve the model version in the registry first, then run:
    python deploy.py --model-package-arn <arn> --role <ARN>
"""

import argparse
import json

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.resources import ModelPackage
from sagemaker.serve import ModelBuilder

ENDPOINT_NAME = "finbert-sentiment-endpoint"


# TODO: Deploy the trained model to a SageMaker managed endpoint
# https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-modelbuilder-creation.html#how-it-works-modelbuilder-creation-deploy
# https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-deploy.html
def deploy_endpoint(role: str, model_package_arn: str, session: Session):
    # TODO: Load the approved model package from the registry and extract
    # the S3 model URI and inference image URI from it.
    registered_package = ModelPackage.get(model_package_name=...)
    s3_uri = ...
    image  = ...

    # TODO: Create a ModelBuilder using the S3 model URI and image URI above.
    model_builder = ModelBuilder()

    # TODO: Build the model

    # TODO: Deploy the model to the endpoint
    predictor = ...
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

    # TODO: Register the endpoint as a scalable target with the specified min and max capacity.

    # TODO: Create a target-tracking scaling policy that maintains an average of 1000 invocations per instance per minute.


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
    parser.add_argument("--model-package-arn", required=True,
                        help="ARN of the approved model package from the SageMaker Model Registry")
    parser.add_argument("--role", default=None)
    args = parser.parse_args()

    session = Session()
    role   = args.role or get_execution_role()
    region = session.boto_region_name

    predictor = deploy_endpoint(role, args.model_package_arn, session)
    configure_autoscaling(predictor.endpoint_name, region)

    verify_endpoint(predictor.endpoint_name, region)

    print(f"\nDeployment complete.")
    print(f"Endpoint: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
