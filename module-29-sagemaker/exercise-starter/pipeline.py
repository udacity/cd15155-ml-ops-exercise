"""
SageMaker Pipeline

Steps:
  1. PreprocessData: downloads & splits financial headlines dataset
  2. TrainModel: fine-tunes FinBERT
  3. EvaluateModel: computes accuracy, writes evaluation.json
  4. CheckAccuracy: ConditionStep: if accuracy >= ACCURACY_THRESHOLD → RegisterModel
  5. RegisterModel: registers model to SageMaker Model Registry

Usage:
    python pipeline.py --role <SageMaker-execution-role-ARN>
"""

import argparse

import boto3
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingS3Input,
    ProcessingOutput,
    ProcessingS3Output,
)
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, SourceCode
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.properties import PropertyFile

MODEL_PACKAGE_GROUP = "FinBERTSentimentClassifiers"

# TODO: Set the minimum accuracy required to register the model
ACCURACY_THRESHOLD = ...


def get_pipeline(role: str, bucket: str, session: PipelineSession, region: str) -> Pipeline:

    # Step 1: Preprocess
    print("Configuring preprocessing step...")
    preprocessor = ScriptProcessor(
        image_uri=image_uris.retrieve(
            framework="sklearn",
            region=region,
            version="1.2-1",
            py_version="py3",
            instance_type="ml.m5.large",
        ),
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
        base_job_name="finbert-preprocess",
    )

    # TODO: Define the ProcessingStep for preprocessing.
    # Run preprocess.py and define three S3 outputs: train, valid, and test.
    # Each output should upload to s3://{bucket}/finbert/processed/<split>
    # and map to /opt/ml/processing/output/<split> inside the container.
    # Hint: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html#step-type-processing
    preprocess_step = ProcessingStep()

    # Step 2: Train
    print("Configuring training step...")
    training_image = image_uris.retrieve(
        framework="huggingface",
        region=region,
        version="4.26.0",
        py_version="py39",
        base_framework_version="pytorch1.13.1",
        image_scope="training",
        instance_type="ml.g4dn.xlarge",
    )

    # TODO: Configure the ModelTrainer using the training image above.
    # Use train.py as the entry script, ml.g4dn.xlarge as the instance type,
    # and pass the train output from preprocess_step as the input data channel.
    # Hint" https://github.com/aws/amazon-sagemaker-examples/blob/default/%20%20%20%20%20%20build_and_train_models/sm-model_trainer/model_trainer_overview.ipynb
    model_trainer = ModelTrainer()

    training_step = TrainingStep(
        name="TrainModel",
        step_args=model_trainer.train(),
    )

    # Step 3: Evaluate 
    print("Configuring evaluation step...")
    evaluator = ScriptProcessor(
        image_uri=image_uris.retrieve(
            framework="huggingface",
            region=region,
            version="4.26.0",
            py_version="py39",
            base_framework_version="pytorch1.13.1",
            image_scope="inference",
            instance_type="ml.m5.large",
        ),
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
        base_job_name="finbert-evaluate",
    )

    # TODO: Define a PropertyFile so the pipeline can read evaluation.json
    # Hint: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html#build-and-manage-propertyfile-property
    evaluation_report = PropertyFile()

    # TODO: Define the ProcessingStep for evaluation.
    # Run evaluate.py with two inputs: model (from training_step) and test (from preprocess_step).
    # Add one output: "evaluation" to s3://{bucket}/finbert/evaluation.
    # Attach evaluation_report to property_files.
    evaluation_step = ProcessingStep()

    # ── Step 4: Register model (if accuracy meets threshold) ──────────────────
    print("Configuring register step...")

    # Retrieve inference image explicitly to avoid internal SDK version lookup issues
    inference_image = image_uris.retrieve(
        framework="huggingface",
        region=region,
        version="4.26.0",
        py_version="py39",
        base_framework_version="pytorch1.13.1",
        image_scope="inference",
        instance_type="ml.g4dn.xlarge",
    )

    # TODO: Build a deployable model from the training step artifacts using ModelBuilder.
    # Use the inference image retrieved above and the model S3 URI from training_step.
    # Hint: https://sagemaker.readthedocs.io/en/v2.208.0/api/inference/model_builder.html
    # Hint: https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-modelbuilder-creation.html
    model_builder = ModelBuilder()

    # TODO: Create a ModelStep that builds the model in SageMaker.
    #Hint: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html#step-type-model

    create_step = ModelStep()

    # TODO: Add a RegisterModel step that registers to the Model Registry
    # Hint https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html#step-type-register-model
    register_step = ModelStep()

    # TODO: Add a ConditionStep that only registers when accuracy >= threshold
    # Hint: https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-types.html#step-type-condition
    condition_step = ConditionStep()

    return Pipeline(
        name="FinBERTPipeline",
        steps=[preprocess_step, training_step, evaluation_step, condition_step],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=None)
    parser.add_argument("--bucket", default=None)
    args = parser.parse_args()

    sagemaker_session = Session()
    pipeline_session = PipelineSession()
    role   = args.role or get_execution_role()
    bucket = args.bucket or sagemaker_session.default_bucket()
    region = sagemaker_session.boto_region_name

    print(f"Role   : {role}")
    print(f"Bucket : {bucket}")

    pipeline = get_pipeline(role=role, bucket=bucket, session=pipeline_session, region=region)
    pipeline.upsert(role_arn=role)

    print("Starting execution...")
    execution = pipeline.start()
    print(f"Execution ARN: {execution.arn}")
    execution.wait()
    print("Done.")


if __name__ == "__main__":
    main()
