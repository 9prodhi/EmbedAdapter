import json
import logging
from typing import List
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException
from utils.types import Message


logger = logging.getLogger(__name__)


def get_chat_completion_old(message: str, max_tokens: int, temperature: float, model: str, top_p: float, bedrock_client: boto3.client):
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

def get_chat_completion(messages: list[Message], max_tokens: int, temperature: float, model: str, top_p: float, bedrock_client: boto3.client):
    result = None
    try:
        system = None
        if messages[0].role == "system":
            system = [{"text": messages.pop(0).content}]

        # system message special handling for mistral 7b instruct
        if model == "mistral.mistral-7b-instruct-v0:2":
            if system:
                messages = [
                    Message(
                        role="user",
                        content="Follow the below instructions for this conversation\n\n"
                        + system[0]["text"],
                    ),
                    Message(role="assistant", content="Sure"),
                ] + messages
                system = None
        converted_messages = [
            {"role": m.role, "content": [{"text": m.content}]} for m in messages
        ]

        payload = dict(
            modelId=model,
            messages=converted_messages,
            system=system,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        )
        if not system:
            payload.pop('system')

        print(f'{payload=}')
        response = bedrock_client.converse(**payload)

        choices = [
            {"index": ind, "message": {
                "role": "assistant", "content": x["text"]}}
            for ind, x in enumerate(response["output"]["message"]["content"])
        ]
        choices[-1]["finish_reason"] = response["stopReason"]

        input_token_count = response["usage"]["inputTokens"]
        output_token_count = response["usage"]["outputTokens"]
        total_token_count = response["usage"]["totalTokens"]

        result = {
            "request_id": response['ResponseMetadata']['RequestId'],
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "total_token_count": total_token_count,
            "response": choices,
            "model": model
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"ClientError for text : {error_code} - {error_message}")
        if error_code == 'ValidationException' and "model identifier is invalid" in error_message:
            logger.error(
                f"Invalid model identifier. Please check if '{model}' is correct and available in your AWS region.")
        raise HTTPException(
            status_code=500, detail=f"Error getting embedding: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error for text : {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error getting embedding: {str(e)}")

    return result
