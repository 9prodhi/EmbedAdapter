import os
import boto3
from botocore.config import Config

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