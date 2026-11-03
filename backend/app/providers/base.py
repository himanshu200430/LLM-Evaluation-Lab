
"""Common interface for all LLM providers, built on LangChain."""

import time
import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@dataclass
class LLMResponse:
    answer: str
    latency_sec: float
    tokens_in: int
    tokens_out: int
    error: str = ""


class BaseProvider:
    """Every provider has a name, a model, prices and an ask() method."""

    name = "base"
    model = ""
    price_in = 0.0
    price_out = 0.0

    def ask(self, question):
        raise NotImplementedError


class LangChainProvider(BaseProvider):
    """Any provider with an OpenAI-style API through LangChain."""

    base_url = ""

    def __init__(self, api_key):
        if not api_key:
            raise ValueError(
                f"API key for {self.name} is missing. Add it to your .env file."
            )

        self.llm = ChatOpenAI(
            model=self.model,
            api_key=api_key,
            base_url=self.base_url,
            temperature=0,
            timeout=60,
            max_retries=0,
        )

    def ask(self, question):
        start = time.time()

        try:
            response = self.llm.invoke([
                SystemMessage(
                    content="Answer in one short sentence or phrase."
                ),
                HumanMessage(content=question),
            ])

            answer = str(response.content).strip()
            usage = response.usage_metadata or {}

            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            error = ""

        except Exception as e:
            answer = ""
            tokens_in = 0
            tokens_out = 0
            error = str(e)

        latency = round(time.time() - start, 2)

        return LLMResponse(
            answer,
            latency,
            tokens_in,
            tokens_out,
            error,
        )

    def ask_batch(self, questions):
        """Send many questions in ONE call.
        Returns (answers, latency_sec, tokens_in, tokens_out, error).
        """

        numbered = "\n".join(
            f"{i + 1}. {q}" for i, q in enumerate(questions)
        )

        start = time.time()

        try:
            response = self.llm.invoke([
                SystemMessage(content=(
                    "Answer each question in one short phrase. Reply ONLY with a JSON array "
                    "of strings, one answer per question, in the same order. No extra text."
                )),
                HumanMessage(content=numbered),
            ])

            text = str(response.content).strip()
            first, last = text.find("["), text.rfind("]")
            if first == -1 or last == -1:
                raise ValueError(f"No JSON list in reply: {text[:100]}")

            answers = json.loads(text)

            if not isinstance(answers, list) or len(answers) != len(questions):
                raise ValueError(
                    f"Expected {len(questions)} answers in a JSON list"
                )

            answers = [str(a).strip() for a in answers]

            usage = response.usage_metadata or {}
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            error = ""

        except Exception as e:
            answers = []
            tokens_in = 0
            tokens_out = 0
            error = str(e)

        latency = round(time.time() - start, 2)

        return answers, latency, tokens_in, tokens_out, error
    def complete(self, system_prompt, user_prompt):
        start = time.time()

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])

            text = str(response.content).strip()

            usage = response.usage_metadata or {}
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)

            error = ""
        
        except Exception as e:
            text = ""
            tokens_in = 0
            tokens_out = 0
            error = str(e)

        return LLMResponse(
            text,
            round(time.time() - start, 2),
            tokens_in,
            tokens_out,
            error,
        )