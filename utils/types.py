from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, List

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
