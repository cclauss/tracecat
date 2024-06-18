import os

from openai import AsyncOpenAI
from outlines import models
from outlines.models.openai import OpenAIConfig


def create_model(model_name: str = "gpt-4o"):
    """Create an LLM model."""
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID"),
        project=os.getenv("OPENAI_PROJECT_ID"),
    )
    config = OpenAIConfig(model_name)
    model = models.openai(client, config)
    return model
