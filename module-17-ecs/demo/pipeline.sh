# Build and push Docker image to ECR

# AWS credentials — paste from AWS Academy > AWS Details > AWS CLI
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

# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json \
  --region $AWS_REGION

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
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0
fi

echo "Using SG: $SG_ID  Subnet: $SUBNET_ID"

# Create ECS service
aws ecs create-service \
  --cluster ner-api-cluster \
  --service-name ner-api-service \
  --task-definition ner-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[$SUBNET_ID],
    securityGroups=[$SG_ID],
    assignPublicIp=ENABLED
  }" \
  --region $AWS_REGION

# Get public IP and verify endpoint
TASK_ARN=$(aws ecs list-tasks --cluster ner-api-cluster \
  --service-name ner-api-service --query taskArns[0] --output text)

ENI_ID=$(aws ecs describe-tasks --cluster ner-api-cluster \
  --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text)

PUBLIC_IP=$(aws ec2 describe-network-interfaces \
  --network-interface-ids $ENI_ID \
  --query 'NetworkInterfaces[0].Association.PublicIp' \
  --output text)

echo "Service running at: http://$PUBLIC_IP:8000"

curl -X POST http://$PUBLIC_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple is headquartered in Cupertino."}'
