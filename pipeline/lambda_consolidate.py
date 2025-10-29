import json
import boto3
import pandas as pd
from io import StringIO

def lambda_handler(event, context):
    """
    Lambda function to consolidate prediction results
    """
    
    s3 = boto3.client('s3')
    bucket_name = event['bucket_name']
    
    try:
        # Use existing consolidation logic from consolidate_prediction_results.ipynb
        y_cols = ['Lower Hutt', 'Petone', 'Wainuiomata', 'Upper Hutt', 'Porirua', 
                  'Wellington High Moa', 'Wellington High Western', 'Wellington Low Level',
                  'North Wellington Moa', 'North Wellington Porirua']
        
        # Get prediction files
        target_files = []
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix='TransformedOutputs/InferenceData/')
        
        for page in page_iterator:
            if 'Contents' in page:
                for content in page['Contents']:
                    if content['Key'].endswith('.csv'):
                        target_files.append(content['Key'])
        
        # Consolidate results (simplified version of your existing logic)
        df_list = []
        for y_col in y_cols:
            # Find prediction file
            target_file = [e for e in target_files if f"/{y_col}.csv" in e]
            if target_file:
                obj = s3.get_object(Bucket=bucket_name, Key=target_file[0])
                data = obj['Body'].read().decode('utf-8')
                df = pd.read_csv(StringIO(data))
                df_list.append(df)
        
        # Combine all results
        if df_list:
            final_df = pd.concat(df_list, axis=1)
            
            # Save to S3
            csv_buffer = StringIO()
            final_df.to_csv(csv_buffer, index=False)
            s3.put_object(
                Bucket=bucket_name, 
                Key='InferenceResults/full_results.csv', 
                Body=csv_buffer.getvalue()
            )
        
        return {
            'statusCode': 200,
            'output_location': f's3://{bucket_name}/InferenceResults/full_results.csv'
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'error': str(e)
        }