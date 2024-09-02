import logging
import time
from typing import Optional, Union, Dict, List
from fastapi import FastAPI, Header, Depends, Security, HTTPException
from fastapi.security.api_key import APIKeyHeader, APIKey, Request
import os
import uvicorn
from starlette.concurrency import run_in_threadpool
from utils.BedrockHandler import BedRockClient
from utils.EmbeddingsHandler import get_embeddings
from utils.ChatHandler import get_chat_completion
from utils.others import process_input_embeddings, build_result_chat
from utils.types import ChatCompletionBody, EmbeddingBody

import os

from dotenv import load_dotenv


load_dotenv()

MODEL_NAME = "amazon.titan-embed-text-v2:0"
API_KEY = os.environ.get("API_KEY", "99")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI()
logger = logging.getLogger(__name__)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")


bedrock_client = BedRockClient()._get_bedrock_client()

@app.post("/v1/embeddings")
async def create_embedding(
    body: EmbeddingBody,
    api_key: APIKey = Depends(get_api_key)
):
    if type(body.input) is str:
        body.input = [body.input]

    texts = [process_input_embeddings(item) for item in body.input]
    start = time.time()


    embeddings = await run_in_threadpool(get_embeddings, texts, bedrock_client, MODEL_NAME)

    logger.info("Embedding generation took %f seconds", time.time() - start)

    return {
        "data": embeddings,  
        "model": MODEL_NAME,
        "object": "list",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

@app.post("/v1/chat/completions")
async def create_embedding(
    body: ChatCompletionBody,
    api_key: APIKey = Depends(get_api_key)
):
    message = body.messages
    max_tokens = body.max_tokens
    temperature = body.temperature
    model = body.model
    top_p = body.top_p

    logger.info(f"message : {message}")
    logger.info(f"max_tokens : {max_tokens}")
    logger.info(f"temperature : {temperature}")
    logger.info(f"model : {model}")
    logger.info(f"top_p : {top_p}")


    start = time.time()
            

    response = await run_in_threadpool(get_chat_completion, message, max_tokens, temperature, model, top_p, bedrock_client)

    logger.info("Chat completion took %f seconds", time.time() - start)

    return build_result_chat(response_json=response)

def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
    )

if __name__ == "__main__":
    main()