#!/bin/sh

export AWS_REGION="us-east-1"
export LAMBDA_FUNCTION_NAME="lambda-stream"
export IMAGE_NAME="lambda-stream"
export IMAGE_TAG="latest"
export REGISTRY_ID=$(aws ecr describe-repositories --query 'repositories[?repositoryName == `'$IMAGE_NAME'`].registryId' --output text)
export IMAGE_URI=${REGISTRY_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}

echo "Docker build..."
docker build -t $IMAGE_URI .

echo "ECR login..."
export login=$(aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${REGISTRY_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com)

if [ "$login" == "Login Succeeded" ] ; then
  echo "Login successful..."
  
  echo "Docker push..."
  docker push $IMAGE_URI:$IMAGE_TAG
  
  echo "Lambda update function code..."
  aws lambda update-function-code --function-name $LAMBDA_FUNCTION_NAME --image-uri $IMAGE_URI:$IMAGE_TAG > /dev/null 2>&1
  sleep 20
  
  echo "Running lambda..."
  aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --payload file://payload.json --cli-binary-format raw-in-base64-out output.txt
  
  echo "Triggering lambda with dynamodb..."
  aws dynamodb put-item --table-name Profiles --item file://item.json --cli-binary-format raw-in-base64-out
  
  echo "Done !!!"
else
  exit 0
fi