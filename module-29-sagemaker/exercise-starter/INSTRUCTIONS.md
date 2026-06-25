# Module 29 - ML Pipelines on AWS SageMaker

## Overview

In this exercise you will build and run an end-to-end ML pipeline on AWS SageMaker. The pipeline preprocesses financial headlines, fine-tunes FinBERT, evaluates accuracy, and conditionally registers the model in the SageMaker Model Registry. You will then deploy the registered model to a managed endpoint, configure auto-scaling, and verify the endpoint.

---

## Prerequisites

Before running any scripts, you need a SageMaker Domain and Studio environment:

1. In the AWS Console, navigate to **SageMaker > Domains > Create domain**
2. Use **Quick setup** and select your default VPC and subnet

Read more: https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html

---

## 1. Complete `pipeline.py`

Complete the TODOs across all four pipeline steps. 

Run the pipeline:

```bash
python pipeline.py --role <SageMaker-execution-role-ARN>
```

Read more: https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html

---

## 2. Complete `deploy.py`

Once the pipeline completes, go to **SageMaker > Model Registry** in the AWS Console, find the latest model version under the `FinBERTSentimentClassifiers` group, and **approve** it. Then copy the model package ARN and run:

```bash
python deploy.py --model-package-arn <arn> --role <ARN>
```

Complete the TODOs to:

- Load the approved model from the Model Registry and deploy it to a SageMaker real-time endpoint
- Configure target-tracking auto-scaling (1 to 4 instances, target 1000 invocations/instance/min)
- Send test requests to verify the endpoint returns predictions

Read more: https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-deploy.html

---

## Teardown

Run the cleanup script to delete all AWS resources created during this exercise:

```bash
python cleanup.py
```

Or with explicit arguments:

```bash
python cleanup.py --bucket <bucket> --region <region>
```

The script deletes resources in this order:
1. Auto-scaling policy and scalable target for the endpoint
2. SageMaker endpoint
3. SageMaker model
4. Model packages and model package group
5. SageMaker pipeline
6. S3 data (`finbert/` prefix in your default bucket)
7. SageMaker Studio apps, user profiles, and domain (including EFS storage)

> **Important:** Always run the cleanup script at the end of the exercise to avoid incurring AWS charges.