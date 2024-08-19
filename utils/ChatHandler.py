import json
import logging
from typing import List
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException


logger = logging.getLogger(__name__)


def get_chat_completion(message: str, max_tokens: int, temperature: float, model: str, top_p: float, bedrock_client: boto3.client):
    result = None
    try:
        raw_body = {
            "prompt" : f"<s>[INST] {message} [/INST]",
            "top_p" : top_p,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body = json.dumps(raw_body)

        response = bedrock_client.invoke_model(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        result=response_body["outputs"][0]["text"]

        input_token_count = response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count']
        output_token_count = response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count']
        total_token_count = input_token_count+output_token_count

        result = {
           "request_id" : response['ResponseMetadata']['RequestId'],
           "input_token_count" : input_token_count,
           "output_token_count" : output_token_count,
           "total_token_count" : total_token_count,
           "response" : result
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"ClientError for text : {error_code} - {error_message}")
        if error_code == 'ValidationException' and "model identifier is invalid" in error_message:
            logger.error(f"Invalid model identifier. Please check if '{model}' is correct and available in your AWS region.")
        raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error for text : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error getting embedding: {str(e)}")
    
    return result