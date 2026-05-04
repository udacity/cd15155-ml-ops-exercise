#!/bin/bash
# Deploy Beans API to AWS ECS with Auto-scaling

# AWS credentials — paste from AWS Academy > AWS Details > AWS CLI
export AWS_ACCESS_KEY_ID="<YOUR_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<YOUR_SECRET_ACCESS_KEY>"
export AWS_SESSION_TOKEN="<YOUR_SESSION_TOKEN>"

AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=beans-api
CLUSTER=beans-api-cluster
SERVICE=beans-api-service

# Create ECR repository
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build, tag and push
docker build -t $ECR_REPO .
docker tag $ECR_REPO:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

# Create ECS cluster
aws ecs create-cluster --cluster-name $CLUSTER --region $AWS_REGION

# Create task execution role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create CloudWatch log group
aws logs create-log-group --log-group-name /ecs/beans-api --region $AWS_REGION

# Render and register task definition
sed -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
    -e "s/\${AWS_REGION}/$AWS_REGION/g" \
    task-definition.json > task-definition-rendered.json

aws ecs register-task-definition \
  --cli-input-json file://task-definition-rendered.json \
  --region $AWS_REGION

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

# Create or update ECS service
SERVICE_STATUS=$(aws ecs describe-services \
  --cluster $CLUSTER --services $SERVICE \
  --query "services[0].status" --output text --region $AWS_REGION)

if [ "$SERVICE_STATUS" = "ACTIVE" ]; then
  aws ecs update-service \
    --cluster $CLUSTER --service $SERVICE \
    --task-definition beans-api --region $AWS_REGION
else
  aws ecs create-service \
    --cluster $CLUSTER \
    --service-name $SERVICE \
    --task-definition beans-api \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region $AWS_REGION
fi

# Configure auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/$CLUSTER/$SERVICE \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 --max-capacity 4 --region $AWS_REGION

SCALE_OUT_ARN=$(aws application-autoscaling put-scaling-policy \
  --policy-name beans-api-scale-out --service-namespace ecs \
  --resource-id service/$CLUSTER/$SERVICE \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type StepScaling \
  --step-scaling-policy-configuration '{"AdjustmentType":"ChangeInCapacity","StepAdjustments":[{"MetricIntervalLowerBound":0,"ScalingAdjustment":1}],"Cooldown":60}' \
  --query PolicyARN --output text --region $AWS_REGION)

SCALE_IN_ARN=$(aws application-autoscaling put-scaling-policy \
  --policy-name beans-api-scale-in --service-namespace ecs \
  --resource-id service/$CLUSTER/$SERVICE \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type StepScaling \
  --step-scaling-policy-configuration '{"AdjustmentType":"ChangeInCapacity","StepAdjustments":[{"MetricIntervalUpperBound":0,"ScalingAdjustment":-1}],"Cooldown":300}' \
  --query PolicyARN --output text --region $AWS_REGION)

aws cloudwatch put-metric-alarm \
  --alarm-name beans-api-cpu-high --metric-name CPUUtilization --namespace AWS/ECS \
  --dimensions Name=ClusterName,Value=$CLUSTER Name=ServiceName,Value=$SERVICE \
  --statistic Average --period 60 --threshold 70 \
  --comparison-operator GreaterThanThreshold --evaluation-periods 1 \
  --alarm-actions $SCALE_OUT_ARN --region $AWS_REGION

aws cloudwatch put-metric-alarm \
  --alarm-name beans-api-cpu-low --metric-name CPUUtilization --namespace AWS/ECS \
  --dimensions Name=ClusterName,Value=$CLUSTER Name=ServiceName,Value=$SERVICE \
  --statistic Average --period 60 --threshold 10 \
  --comparison-operator LessThanThreshold --evaluation-periods 5 \
  --alarm-actions $SCALE_IN_ARN --region $AWS_REGION

echo "Deployment complete. Run test.sh to verify the endpoint."
