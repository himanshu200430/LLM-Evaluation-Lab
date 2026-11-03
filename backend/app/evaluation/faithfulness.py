import json

from app.evaluation.prompts import FAITHFULNESS_SYSTEM, build_faithfulness_input
from app.schemas.judge import FaithfulnessResult

def parse_faithfulness_reply(text):
    first = text.find("{")
    last = text.rfind("}")

    if first == -1 or last == -1:
        raise ValueError(f"no JSON object in reply: {text[:80]}")

    return FaithfulnessResult(
        **json.loads(text[first:last + 1])
    )

def judge_faithfulness(provider, question, context, answer):
    error = ""

    for _ in range(2):
        response = provider.complete(
            FAITHFULNESS_SYSTEM,
            build_faithfulness_input(question, context, answer),
        )
        if response.error:
            return None, response.error
        
        try:
            return parse_faithfulness_reply(response.answer), ""

        except ValueError as e:
            error = f"bad judge output: {str(e)[:100]}"

    return None, error