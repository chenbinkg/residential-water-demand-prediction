#!/bin/bash

# Water Demand Prediction Pipeline Deployment Script

REGION="ap-southeast-2"  # Adjust as needed
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET_NAME="niwa-water-demand-modelling"

echo "Deploying Water Demand Prediction Pipeline..."

# 1. Create Lambda functions
echo "Creating Lambda functions..."

# Data Prep Lambda
aws lambda create-function \
    --function-name water-demand-data-prep \
    --runtime python3.9 \
    --role arn:aws:iam::${ACCOUNT_ID}:role/AmazonSageMaker-ExecutionRole-20240711T130963 \
    --handler lambda_data_prep.lambda_handler \
    --zip-file fileb://lambda_data_prep.zip \
    --timeout 900 \
    --memory-size 1024

# Inference Lambda
aws lambda create-function \
    --function-name water-demand-inference \
    --runtime python3.9 \
    --role arn:aws:iam::${ACCOUNT_ID}:role/AmazonSageMaker-ExecutionRole-20240711T130963 \
    --handler lambda_inference.lambda_handler \
    --zip-file fileb://lambda_inference.zip \
    --timeout 900 \
    --memory-size 512

# Consolidate Lambda
aws lambda create-function \
    --function-name water-demand-consolidate \
    --runtime python3.9 \
    --role arn:aws:iam::${ACCOUNT_ID}:role/AmazonSageMaker-ExecutionRole-20240711T130963 \
    --handler lambda_consolidate.lambda_handler \
    --zip-file fileb://lambda_consolidate.zip \
    --timeout 900 \
    --memory-size 1024

# 2. Create Step Functions state machine
echo "Creating Step Functions state machine..."
aws stepfunctions create-state-machine \
    --name water-demand-pipeline \
    --definition file://step_functions_definition.json \
    --role-arn arn:aws:iam::${ACCOUNT_ID}:role/AmazonSageMaker-ExecutionRole-20240711T130963

# 3. Create EventBridge rule
echo "Creating EventBridge rule..."
aws events put-rule \
    --name water-demand-s3-upload \
    --event-pattern file://eventbridge_pattern.json \
    --state ENABLED

# Add target to rule
aws events put-targets \
    --rule water-demand-s3-upload \
    --targets Id=1,Arn=arn:aws:states:${REGION}:${ACCOUNT_ID}:stateMachine:water-demand-pipeline,RoleArn=arn:aws:iam::${ACCOUNT_ID}:role/AmazonSageMaker-ExecutionRole-20240711T130963

echo "Pipeline deployment complete!"
echo "Upload files to s3://${BUCKET_NAME}/uploads/ to trigger the pipeline"