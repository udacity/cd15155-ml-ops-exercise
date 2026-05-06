"""
Demo: 3-step SageMaker Pipeline — Preprocess → Train (FinBERT) → Evaluate

Steps:
  1. PreprocessData  — downloads & splits the financial headlines dataset
  2. TrainModel      — fine-tunes FinBERT (GPU required: ml.g4dn.xlarge)
  3. EvaluateModel   — computes accuracy on the held-out test split

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
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, SourceCode


def get_pipeline(role: str, bucket: str, session: PipelineSession, region: str) -> Pipeline:

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
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

    preprocess_step = ProcessingStep(
        name="PreprocessData",
        step_args=preprocessor.run(
            code="preprocess.py",
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/finbert/processed/train",
                        local_path="/opt/ml/processing/output/train",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
                ProcessingOutput(
                    output_name="valid",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/finbert/processed/valid",
                        local_path="/opt/ml/processing/output/valid",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
                ProcessingOutput(
                    output_name="test",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/finbert/processed/test",
                        local_path="/opt/ml/processing/output/test",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
        ),
    )

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print("Configuring training step...")
    model_trainer = ModelTrainer(
        training_image=image_uris.retrieve(
            framework="huggingface",
            region=region,
            version="4.26.0",
            py_version="py39",
            base_framework_version="pytorch1.13.1",
            image_scope="training",
            instance_type="ml.g4dn.xlarge",
        ),
        source_code=SourceCode(source_dir=".", entry_script="train.py"),
        compute=Compute(
            instance_type="ml.g4dn.xlarge",
            instance_count=1,
        ),
        base_job_name="finbert-train",
        sagemaker_session=session,
        role=role,
        input_data_config=[
            InputData(
                channel_name="train",
                data_source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
            ),
        ],
    )

    training_step = TrainingStep(
        name="TrainModel",
        step_args=model_trainer.train(),
    )

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    print("Configuring evaluation step...")
    evaluator = ScriptProcessor(
        image_uri=image_uris.retrieve(
            framework="pytorch",
            region=region,
            version="2.0.0",
            py_version="py310",
            image_scope="training",
            instance_type="ml.m5.large",
        ),
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
        base_job_name="finbert-evaluate",
    )

    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        step_args=evaluator.run(
            code="evaluate.py",
            inputs=[
                ProcessingInput(
                    input_name="model",
                    s3_input=ProcessingS3Input(
                        s3_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                        local_path="/opt/ml/processing/model",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                ),
                ProcessingInput(
                    input_name="test",
                    s3_input=ProcessingS3Input(
                        s3_uri=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                            "test"
                        ].S3Output.S3Uri,
                        local_path="/opt/ml/processing/test",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{bucket}/finbert/evaluation",
                        local_path="/opt/ml/processing/evaluation",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
        ),
    )

    return Pipeline(
        name="FinBERTPipeline",
        steps=[preprocess_step, training_step, evaluation_step],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN (auto-detected in Studio)")
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

    print("\nUpserting pipeline...")
    pipeline.upsert(role_arn=role)

    print("Starting execution...")
    execution = pipeline.start()
    print(f"Execution ARN: {execution.arn}")
    print("\nWaiting for completion...")
    execution.wait()
    print("Done.")
    print(f"\nEvaluation: s3://{bucket}/finbert/evaluation/evaluation.json")


if __name__ == "__main__":
    main()
