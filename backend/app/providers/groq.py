from app.config import GROQ_API_KEY, GROQ_MODEL
from app.providers.base import LangChainProvider

class GroqProvider(LangChainProvider):
    base_url = "https://api.groq.com/openai/v1"
    price_in = 0.10
    price_out = 0.50

    def __init__(self, model=GROQ_MODEL):
        self.model = model
        self.name = f"groq/{model}"
        super().__init__(api_key=GROQ_API_KEY)
        