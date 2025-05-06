import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


def get_client() -> InferenceClient:
    load_dotenv()
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=os.getenv("HF_ACCESS")
    )
    return client
