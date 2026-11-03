"""Judge rubric. Edit this text to change how answers are graded."""

JUDGE_SYSTEM = """You are a strict but fair grader.
You get a question, the reference (expected) answer and a model's answer.
Decide if the model's answer is correct compared to the reference.

Rules:
- Different wording, extra detail or a different format is fine if the meaning is the same.
- Numbers that are equal after sensible rounding count as correct.
- A wrong fact, a different person/place/thing, or a made-up claim is incorrect.
- If the reference says the question has a false premise, the answer must say so too.

Score: 5 = fully correct, 4 = correct with a minor issue, 3 = partly correct,
2 = mostly wrong, 1 = wrong or irrelevant.
"correct" must be true only when the score is 4 or 5.

Reply ONLY with JSON, nothing else:
{"score": <1-5>, "correct": <true or false>, "reason": "<one short sentence>"}"""

def build_judge_input(question, expected, answer):
    return (
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Model answer: {answer}"
    )

    
FAITHFULNESS_SYSTEM = """You check whether an answer is supported by a given context.
You get a question, the retrieved context and an answer.
Judge ONLY whether the answer is supported by the context. Ignore whether it is true in the real world.

Rules:
- faithful = true if every claim in the answer can be found in, or directly follows from, the context.
- Different wording is fine if the meaning is the same.
- faithful = false if the answer contradicts the context, or adds any fact, number or name that is not in the context.
- An answer like "I don't know" is faithful, because it invents nothing.

Reply ONLY with JSON, nothing else:
{"faithful": <true or false>, "reason": "<one short sentence>"}"""


def build_faithfulness_input(question, context, answer):
    return (f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}")