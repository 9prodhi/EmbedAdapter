# Private Embedding Server API

Drop in replacement for OpenAI's embedding API. OpenAI-compatible interface.
```
from openai import OpenAI

# Create a custom client
client = OpenAI(
    api_key="MYKEY",
    base_url="http://localhost:8001/v1"
)

# Create embeddings
response = client.embeddings.create(
    input=["Namaste llm"],
    model="nomic-embed-text"
)

# Print the response
print(response)

# If you want to access the embedding directly, you can do:
embedding = response.data[0].embedding
print(f"Embedding (first 5 values): {embedding[:5]}")
```

Also via CURL

```
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer MYKEY" \
  -d '{
    "input": "Your text string goes here",
    "model": "nomic-embed-text"
  }'
```

### Start Server

```
$ python server.py
```


## Installation

conda create -n embed-dev python=3.12
conda activate embed-dev
pip install fastapi
pip install uvicorn
pip install openai
pip install boto3 python-dotenv

## .env file fow AWS Bedrock
```
AWS_ACCESS_KEY_ID=999
AWS_SECRET_ACCESS_KEY=9999
AWS_REGION=us-east-1  # Optional, defaults to us-east-1

```

## Credits:
https://github.com/toshsan/embedding-server





