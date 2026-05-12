#!/bin/bash
# Deploy Beans API to AWS ECS with Auto-scaling

# AWS credentials
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_SESSION_TOKEN=""

AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=beans-api
CLUSTER=beans-api-cluster
SERVICE=beans-api-service

# TODO Create ECR repository

# TODO Authenticate Docker to ECR

# TODO Build, tag and push image to ECR


# TODO Create ECS cluster

# Create task execution role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create CloudWatch log group
aws logs create-log-group --log-group-name /ecs/beans-api --region $AWS_REGION

# TODO Render task-definition.json by substituting AWS_ACCOUNT_ID and AWS_REGION,
# then register the rendered task definition

# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
  --query "Subnets[0].SubnetId" --output text)

# Get existing security group or create it
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=beans-api-sg" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text)

if [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name beans-api-sg \
    --description "Security group for Beans API ECS service" \
    --vpc-id $VPC_ID \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
fi

echo "Using SG: $SG_ID  Subnet: $SUBNET_ID"

# TODO Create or update ECS service


# TODO Configure auto-scaling

# TODO Create CloudWatch alarms