#!/bin/bash
# Tear down all AWS resources created by deploy.sh

export AWS_PAGER=""

AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=ner-api
CLUSTER=ner-api-cluster
SERVICE=ner-api-service

echo "Starting cleanup..."

# Scale down and delete ECS service
echo "Deleting ECS service..."
aws ecs update-service \
  --cluster $CLUSTER --service $SERVICE \
  --desired-count 0 \
  --region $AWS_REGION || true

aws ecs delete-service \
  --cluster $CLUSTER --service $SERVICE \
  --force \
  --region $AWS_REGION || true

# Deregister all task definition revisions
echo "Deregistering task definitions..."
TASK_DEFS=$(aws ecs list-task-definitions \
  --family-prefix ner-api \
  --query "taskDefinitionArns[]" --output text --region $AWS_REGION)

for td in $TASK_DEFS; do
  aws ecs deregister-task-definition --task-definition $td --region $AWS_REGION || true
done

# Delete ECS cluster
echo "Deleting ECS cluster..."
aws ecs delete-cluster --cluster $CLUSTER --region $AWS_REGION || true

# Delete CloudWatch log group
echo "Deleting CloudWatch log group..."
aws logs delete-log-group --log-group-name /ecs/ner-api --region $AWS_REGION || true

# Detach policy and delete IAM role
echo "Deleting IAM role..."
aws iam detach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy || true

aws iam delete-role --role-name ecsTaskExecutionRole || true

# Delete security group
echo "Deleting security group..."
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=ner-api-sg" \
  --query "SecurityGroups[0].GroupId" --output text)

if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ]; then
  aws ec2 delete-security-group --group-id $SG_ID || true
fi

# Delete ECR repository and all images
echo "Deleting ECR repository..."
aws ecr delete-repository \
  --repository-name $ECR_REPO \
  --force \
  --region $AWS_REGION || true

echo "Cleanup complete."
