#!/bin/bash
# Deploy NER API to AWS ECS

# AWS credentials
export AWS_ACCESS_KEY_ID="<YOUR_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<YOUR_SECRET_ACCESS_KEY>"
export AWS_SESSION_TOKEN="<YOUR_SESSION_TOKEN>"

AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=ner-api
IMAGE_TAG=latest

# Create ECR repository
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build, tag and push
docker build -t $ECR_REPO .
docker tag $ECR_REPO:$IMAGE_TAG \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

# Create ECS cluster
aws ecs create-cluster --cluster-name ner-api-cluster --region $AWS_REGION

# Create task execution role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json \
  --region $AWS_REGION

# Create CloudWatch log group
aws logs create-log-group --log-group-name /ecs/ner-api --region $AWS_REGION

# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
  --query "Subnets[0].SubnetId" --output text)

# Get existing security group or create it
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=ner-api-sg" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text)

if [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name ner-api-sg \
    --description "Security group for NER API ECS service" \
    --vpc-id $VPC_ID \
    --query GroupId --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0
fi

echo "Using SG: $SG_ID  Subnet: $SUBNET_ID"

# Create or update ECS service
SERVICE_STATUS=$(aws ecs describe-services \
  --cluster ner-api-cluster --services ner-api-service \
  --query "services[0].status" --output text --region $AWS_REGION)

if [ "$SERVICE_STATUS" = "ACTIVE" ]; then
  aws ecs update-service \
    --cluster ner-api-cluster \
    --service ner-api-service \
    --task-definition ner-api \
    --region $AWS_REGION
else
  aws ecs create-service \
    --cluster ner-api-cluster \
    --service-name ner-api-service \
    --task-definition ner-api \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region $AWS_REGION
fi

echo "Deployment complete. Run test.sh to verify the endpoint."
