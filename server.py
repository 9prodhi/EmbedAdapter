import logging
import time
from typing import Optional, Union, Dict, List
from fastapi import FastAPI, Header, Depends, Request
from pydantic import BaseModel, Field
import os
import uvicorn
from starlette.concurrency import run_in_threadpool
from utils.BedrockHandler import BedRockClient
from utils.EmbeddingsHandler import get_embeddings
from utils.ChatHandler import get_chat_completion
from utils.others import process_input_embeddings, build_result_chat


import os

from dotenv import load_dotenv


load_dotenv()


app = FastAPI()
logger = logging.getLogger(__name__)

MODEL_NAME = "amazon.titan-embed-text-v2:0"

class EmbeddingBody(BaseModel):
    input: str | List[Union[str, Dict[str, str]]] = Field(description="List of strings or key-value pairs for embedding")
    # input: Optional[str | list[str]]
    model: Optional[str] = Field(
        default=None, title="model name. not in use", max_length=300
    )

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionBody(BaseModel):
    messages: list[Message]
    model: str
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0
    top_p: Optional[float] = 1


bedrock_client = BedRockClient()._get_bedrock_client()

@app.post("/v1/embeddings")
async def create_embedding(
    body: EmbeddingBody,
    Authorization: Optional[str] = Header(None)
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
    Authorization: Optional[str] = Header(None)
):
    message = body.messages[0].content
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