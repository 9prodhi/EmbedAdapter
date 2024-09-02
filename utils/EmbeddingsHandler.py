import json
import logging
from typing import List
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException


logger = logging.getLogger(__name__)


def get_embeddings(texts: List[str], bedrock_client: boto3.client, model: str):
    embeddings = []
    for index, text in enumerate(texts):
        try:
            body = json.dumps({"inputText": text})
            response = bedrock_client.invoke_model(
                body=body,
                modelId=model,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get("embedding")
            if embedding is None:
                logger.error(f"No embedding found in response for text at index {index}")
                logger.error(f"Response body: {response_body}")
                raise ValueError("No embedding found in response")
            embeddings.append({
                "embedding": embedding,
                "index": index,
                "object": "embedding"
            })
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"ClientError for text at index {index}: {error_code} - {error_message}")
            if error_code == 'ValidationException' and "model identifier is invalid" in error_message:
                logger.error(f"Invalid model identifier. Please check if '{model}' is correct and available in your AWS region.")
            raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error for text at index {index}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error getting embedding: {str(e)}")
    
    return embeddings