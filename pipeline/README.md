# Water Demand Prediction Pipeline

Automated pipeline for water demand prediction using AWS Step Functions, Lambda, and SageMaker.

## Architecture

```
S3 Upload → EventBridge → Step Functions → [Data Prep → Inference → Consolidation] → Results
```

## Components

1. **EventBridge Rule**: Triggers on S3 object creation in `uploads/` folder
2. **Step Functions**: Orchestrates the pipeline workflow
3. **Lambda Functions**:
   - `lambda_data_prep.py`: Processes input data using existing prep logic
   - `lambda_inference.py`: Triggers SageMaker batch transform jobs
   - `lambda_consolidate.py`: Consolidates results into final output

## Deployment

1. **Package Lambda functions**:
```bash
cd pipeline
zip lambda_data_prep.zip lambda_data_prep.py
zip lambda_inference.zip lambda_inference.py  
zip lambda_consolidate.zip lambda_consolidate.py
```

2. **Deploy pipeline**:
```bash
chmod +x deploy.sh
./deploy.sh
```

## Usage

1. **Upload file to trigger pipeline**:
```bash
aws s3 cp your_data.csv s3://niwa-water-demand-modelling/uploads/
```

2. **Monitor execution**:
```bash
aws stepfunctions list-executions --state-machine-arn arn:aws:states:REGION:ACCOUNT:stateMachine:water-demand-pipeline
```

3. **Download results**:
```bash
aws s3 cp s3://niwa-water-demand-modelling/InferenceResults/full_results.csv ./
```

## Benefits vs Manual Process

- **Automated**: No manual notebook execution
- **Event-driven**: Triggers automatically on file upload
- **Scalable**: Handles multiple concurrent requests
- **Monitored**: Built-in error handling and retry logic
- **Cost-effective**: Pay only for execution time

## Customization

- Modify `step_functions_definition.json` for different workflow logic
- Update Lambda functions for custom processing requirements
- Adjust EventBridge pattern for different trigger conditions