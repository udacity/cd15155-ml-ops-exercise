"""
Demo: 3-step SageMaker Pipeline — Preprocess → Train (FinBERT) → Evaluate

Steps:
  1. PreprocessData  — downloads & splits the financial headlines dataset
  2. TrainModel      — fine-tunes FinBERT (GPU required: ml.g4dn.xlarge)
  3. EvaluateModel   — computes accuracy on the held-out test split

Usage (from SageMaker Studio terminal — role is auto-detected):
    python pipeline.py

Usage (from local machine):
    python pipeline.py --role <SageMaker-execution-role-ARN>
"""

import argparse

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorchProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


def get_pipeline(role: str, bucket: str, session: sagemaker.Session) -> Pipeline:

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    print("Configuring preprocessing step...")
    preprocessor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=session,
        base_job_name="finbert-preprocess",
    )

    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=preprocessor,
        code="preprocess.py",
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/finbert/processed/train",
            ),
            ProcessingOutput(
                output_name="valid",
                source="/opt/ml/processing/output/valid",
                destination=f"s3://{bucket}/finbert/processed/valid",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/finbert/processed/test",
            ),
        ],
    )

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print("Configuring training step...")
    estimator = HuggingFace(
        entry_point="train.py",
        role=role,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        transformers_version="4.26.0",
        pytorch_version="1.13.1",
        py_version="py39",
        sagemaker_session=session,
    )

    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri
            ),
            "valid": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "valid"
                ].S3Output.S3Uri
            ),
        },
    )

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    print("Configuring evaluation step...")
    evaluator = PyTorchProcessor(
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="2.0",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="finbert-evaluate",
    )

    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluator,
        code="evaluate.py",
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/finbert/evaluation",
            )
        ],
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

    session = sagemaker.Session(boto_session=boto3.Session())
    role   = args.role or sagemaker.get_execution_role()
    bucket = args.bucket or session.default_bucket()

    print(f"Role   : {role}")
    print(f"Bucket : {bucket}")

    pipeline = get_pipeline(role=role, bucket=bucket, session=session)

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
