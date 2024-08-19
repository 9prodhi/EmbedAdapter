from typing import Optional, Union, Dict, List

def process_input_embeddings(item: Union[str, Dict[str, str]]) -> str:
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        key, value = next(iter(item.items()))
        return f"{key}: {value}"
    else:
        raise ValueError(f"Unexpected input type: {type(item)}")
    


def build_result_chat(response_json):
    res = {
        "id": response_json['request_id'],
        "object": "chat.completion",
        "created": 1723136983,
        "model": "mistral",
        "system_fingerprint": "fp_ollama",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_json['response']
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": response_json['input_token_count'],
            "completion_tokens": response_json['output_token_count'],
            "total_tokens": response_json['total_token_count']
        }
    }
    return res