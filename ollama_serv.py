import logging
import time
from typing import Optional, Union, Dict, List
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import os
import uvicorn
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

import httpx



app = FastAPI()
logger = logging.getLogger(__name__)

class EmbeddingBody(BaseModel):
    input: List[Union[str, Dict[str, str]]] = Field(description="List of strings or key-value pairs for embedding")
    model: Optional[str] = Field(
        default=None, title="model name. not in use", max_length=300
    )



def process_input(item: Union[str, Dict[str, str]]) -> str:
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        key, value = next(iter(item.items()))
        return f"{key}: {value}"
    else:
        raise ValueError(f"Unexpected input type: {type(item)}")


OLLAMA_BASE_URL = "http://localhost:11434/v1"  
MODEL_NAME = "nomic-embed-text"

async def get_embeddings_ollama(texts: List[str], model: str = MODEL_NAME):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/embeddings",
                json={"model": model, "input": texts}
            )
            response.raise_for_status()
            response_data = response.json()
            
            if "data" not in response_data or not response_data["data"]:
                raise ValueError("No embedding data found in response")
            
            embeddings = [
                {
                    "embedding": item["embedding"],
                    "index": item["index"],
                    "object": "embedding"
                }
                for item in response_data["data"]
            ]
            
            # Extract the correct token usage from the response
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            return {
                "data": embeddings,
                "model": MODEL_NAME,
                "object": "list",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens
                }
            }
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Error getting embedding: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error getting embedding: {str(e)}")



@app.post("/v1/embeddings")
async def create_embedding(
    body: EmbeddingBody,
    Authorization: Optional[str] = Header(None)
):
    texts = [process_input(item) for item in body.input]
    start = time.time()

    # Use thread pooling for the API call
    # embeddings = await run_in_threadpool(get_embeddings_ollama, texts)
    embeddings = await get_embeddings_ollama(texts)

    logger.info("Embedding generation took %f seconds", time.time() - start)

    return embeddings

def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
    )

if __name__ == "__main__":
    main()