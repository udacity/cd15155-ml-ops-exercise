# Exercise: Deploy a Model API and Apply Auto-scaling on AWS ECS

## Overview
Deploy the beans disease classifier API to AWS ECS, verify it serves correct predictions, then configure auto-scaling to handle traffic bursts.

You can complete this exercise using the command line interface or the Console.

---

## Setup

Get your credentials from and configure the CLI:

```bash
aws configure
```

Then set the variables used throughout the exercise:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=beans-api
export CLUSTER=beans-api-cluster
export SERVICE=beans-api-service
```

---

## 1: Build the Docker image and push it to ECR

Create an ECR repository, authenticate Docker to it, then build, tag and push the image.

AWS documentation: https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html

If you prefer to use the command line, complete the relevant TODOs in `deploy.sh`.


## 2: Create ECS Cluster and Service

- Create an ECS cluster named `beans-api-cluster`

- Create a task execution role so ECS can pull images from ECR and write logs to CloudWatch (commands provided in `deploy.sh`)

- Create a dedicated CloudWatch log group (command provided in `deploy.sh`)

- The provided `task-definition.json` contains placeholders for your account ID and region. Substitute them before registering the task definition.

- A security group allowing inbound traffic on port 8000 is needed. The commands to create it are provided in `deploy.sh`.

- Create the ECS service using Fargate with 1 desired task.

Read more about to create ECS service using the Console [here](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/getting-started-fargate.html)

---

## 3: Add autoscaling to your service

Register the ECS service as a scalable target with a minimum of 1 task and a maximum of 4 tasks.

Create two **Step Scaling** policies (not Target Tracking):
- A **scale-out** Step Scaling policy that adds 1 task when CPU utilization is greater than or equal to 70%.
- A **scale-in** Step Scaling policy that removes 1 task when CPU utilization is less than or equal to 10% for 5 consecutive minutes.

If you prefer to use the command line, complete the auto-scaling TODO in `deploy.sh`.

Follow the instructions in the [AWS documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/step-scaling-create-policy.html).
Another useful resource is available [here](https://docs.aws.amazon.com/autoscaling/application/userguide/create-step-scaling-policy-cli.html)

---

## 4: Add CloudWatch alarms

CloudWatch alarms connect a metric to a scaling policy. Each alarm watches CPU utilization on your ECS service and fires the corresponding policy when the threshold is crossed.

- The **scale-out alarm** triggers the scale-out policy when CPU > 70% for 1 consecutive minute.
- The **scale-in alarm** triggers the scale-in policy when CPU < 10% for 5 consecutive minutes (longer cooldown to avoid premature scale-in).

Both alarms need the policy ARNs from step 3 as their alarm actions.

If you prefer to use the command line, complete the alarm TODO in `deploy.sh`.

Console: navigate to **CloudWatch > Alarms > Create alarm**, select the ECS CPUUtilization metric filtered by your cluster and service.

For more information: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/US_AlarmAtThresholdEC2.html

---

## 5: Verify your service

Run `test.sh` to confirm the service is up and returns correct predictions:

```bash
sh test.sh
```

---

## Teardown

Run the cleanup script to delete all AWS resources created during this exercise:

```bash
bash cleanup.sh
```

The script deletes resources in this order:
1. CloudWatch alarms
2. Auto-scaling policies and scalable target
3. ECS service (scaled down to 0 first, then deleted)
4. Task definition revisions
5. ECS cluster
6. CloudWatch log group
7. IAM role and policy attachment
8. Security group
9. ECR repository and all images

> **Important:** Always run the cleanup script at the end of the exercise to avoid incurring AWS charges.
