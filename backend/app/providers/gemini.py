
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.providers.base import LangChainProvider


class GeminiProvider(LangChainProvider):
    base_url = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    price_in = 0.30 
    price_out = 2.50

    def __init__(self):
        self.model = GEMINI_MODEL
        self.name = GEMINI_MODEL 

        super().__init__(api_key=GEMINI_API_KEY)
