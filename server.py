import logging
import time
from typing import Optional, Union, Dict, List
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import os
import uvicorn
from starlette.concurrency import run_in_threadpool

import json
import os
import boto3
import botocore
from botocore.config import Config
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

MODEL_NAME = "amazon.titan-embed-text-v2:0"
app = FastAPI()
logger = logging.getLogger(__name__)


class EmbeddingBody(BaseModel):
    input: List[Union[str, Dict[str, str]]] = Field(description="List of strings or key-value pairs for embedding")
    model: Optional[str] = Field(
        default=None, title="model name. not in use", max_length=300
    )

class BedRockClient:
    def __init__(self):
        self.module = "AWS Bedrock"
        self.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.environ.get('AWS_REGION', 'us-east-1')  # Default to us-east-1 if not specified

    def _get_bedrock_client(self, runtime=True):
        """
        Get the Bedrock client
        :param runtime: True for bedrock-runtime, False for bedrock
        :return: Bedrock client
        """
        retry_config = Config(
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        )

        service_name = 'bedrock-runtime' if runtime else 'bedrock'

        bedrock_client = boto3.client(
            service_name=service_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region,
            config=retry_config
        )

        return bedrock_client

def process_input(item: Union[str, Dict[str, str]]) -> str:
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        key, value = next(iter(item.items()))
        return f"{key}: {value}"
    else:
        raise ValueError(f"Unexpected input type: {type(item)}")

# def get_embeddings(texts: List[str], model: str):
#     return client.embeddings.create(input=texts, model=model).data

bedrock_client = BedRockClient()._get_bedrock_client()

def get_embeddings(texts: List[str], model: str = MODEL_NAME):
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


@app.post("/v1/embeddings")
async def create_embedding(
    body: EmbeddingBody,
    Authorization: Optional[str] = Header(None)
):
    texts = [process_input(item) for item in body.input]
    start = time.time()


    embeddings = await run_in_threadpool(get_embeddings, texts)

    logger.info("Embedding generation took %f seconds", time.time() - start)

    return {
        "data": embeddings,  
        "model": MODEL_NAME,
        "object": "list",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
    )

if __name__ == "__main__":
    main()