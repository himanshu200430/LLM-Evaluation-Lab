import json

from app.evaluation.prompts import JUDGE_SYSTEM, build_judge_input
from app.schemas.judge import JudgeResult

def parse_judge_reply(text):
    first = text.find("{")
    last = text.rfind("}")

    if first == -1 or last == -1:
        raise ValueError(f"no JSON object in reply: {text[:80]}")

    return JudgeResult(**json.loads(text[first:last + 1]))

def judge_answer(provider, question, expected, answer):
    error = ""

    for _ in range(2):
        response = provider.complete(
            JUDGE_SYSTEM,
            build_judge_input(question, expected, answer),
        )

        if response.error:
            return None, response.error

        try:
            return parse_judge_reply(response.answer), ""
        
        except ValueError as e:
            error = f"bad judge output: {str(e)[:100]}"

    return None, error
