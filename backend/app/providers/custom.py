"""A model you add yourself: any provider with an OpenAI-compatible API."""

import os

from app import config  # noqa: F401 (importing config loads the .env file)
from app.providers.base import LangChainProvider


class CustomProvider(LangChainProvider):
    def __init__(
        self,
        name,
        model,
        base_url,
        api_key_env="",
        price_in=0.0,
        price_out=0.0,
    ):
        self.name = name
        self.model = model
        self.base_url = base_url

        # USD per 1M input and output tokens
        self.price_in = float(price_in)
        self.price_out = float(price_out)

        if api_key_env.strip():
            api_key = os.getenv(api_key_env.strip(), "")

            if not api_key:
                raise ValueError(
                    f"{name}: the variable {api_key_env.strip()} is empty. "
                    "Add it to your .env file."
                )

        else:
            # Local servers such as Ollama need no API key
            api_key = "not-needed"

        super().__init__(api_key=api_key)