import json
import boto3
from time import sleep

def lambda_handler(event, context):
    """
    Lambda function to trigger SageMaker batch transform jobs
    """
    
    sagemaker = boto3.client('sagemaker')
    
    # Model mappings from your existing code
    auto_ml_job_dict = {
        'NorthWellingtonMoa': 'Canvas1734649444174-trial-t1-1',
        'WellingtonLowLevel': 'Canvas1734648978161-trial-t1-1',
        'Petone': 'Canvas1733434154045-trial-t1-1',
        'WellingtonHighWestern': 'Canvas1733085655509-trial-t1-1',
        'WellingtonHighMoa': 'Canvas1733372214860-trial-t1-1',
        'NorthWellingtonPorirua': 'Canvas1733369877242-trial-t1-1',
        'Porirua': 'Canvas1733437572452-trial-t1-1',
        'Wainuiomata': 'Canvas1734649248674-trial-t1-1',
        'UpperHutt': 'Canvas1734649294393-trial-t1-1',
        'LowerHutt': 'Canvas1734649384856-trial-t1-1'
    }
    
    processed_files = event['processed_files']
    bucket_name = event['bucket_name']
    
    job_names = []
    
    try:
        for file_path in processed_files:
            # Extract site name from path
            site_name = file_path.split('/')[1]
            model_name = auto_ml_job_dict.get(site_name)
            
            if model_name:
                # Create transform job
                job_name = f"{model_name}-{context.aws_request_id[:8]}"
                
                sagemaker.create_transform_job(
                    TransformJobName=job_name,
                    ModelName=model_name,
                    TransformInput={
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': f's3://{bucket_name}/TransformedInputs/{file_path}'
                            }
                        },
                        'ContentType': 'text/csv',
                        'SplitType': 'Line'
                    },
                    TransformOutput={
                        'S3OutputPath': f's3://{bucket_name}/TransformedOutputs/InferenceData/{site_name}',
                        'AssembleWith': 'Line'
                    },
                    TransformResources={
                        'InstanceType': 'ml.c5.xlarge',
                        'InstanceCount': 1
                    }
                )
                
                job_names.append(job_name)
        
        return {
            'statusCode': 200,
            'job_names': job_names,
            'bucket_name': bucket_name
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'error': str(e)
        }