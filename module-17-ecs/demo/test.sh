# Verify the NER API endpoint

# AWS credentials
export AWS_ACCESS_KEY_ID="<YOUR_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<YOUR_SECRET_ACCESS_KEY>"
export AWS_SESSION_TOKEN="<YOUR_SESSION_TOKEN>"

AWS_REGION=us-east-1

# Get public IP of the running task
TASK_ARN=$(aws ecs list-tasks \
  --cluster ner-api-cluster \
  --service-name ner-api-service \
  --query taskArns[0] --output text --region $AWS_REGION)

ENI_ID=$(aws ecs describe-tasks \
  --cluster ner-api-cluster --tasks $TASK_ARN --region $AWS_REGION \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text)

PUBLIC_IP=$(aws ec2 describe-network-interfaces \
  --network-interface-ids $ENI_ID \
  --query 'NetworkInterfaces[0].Association.PublicIp' \
  --output text)

echo "Service running at: http://$PUBLIC_IP:8000"

# Health check
curl http://$PUBLIC_IP:8000/health

# Predict
curl -X POST http://$PUBLIC_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple is headquartered in Cupertino."}'