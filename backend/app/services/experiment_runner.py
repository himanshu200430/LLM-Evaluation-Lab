import time

from app.evaluation.exact_match import contains_match, exact_match, normalized_match
from app.services.cost_calculator import estimate_cost

def run_model(items, provider, delay_sec=0.5):
    results = []

    for i, item in enumerate(items):
        response = provider.ask(item.question)
        failed = bool(response.error)

        results.append({
            "model": provider.name,
            "question": item.question,
            "expected_answer": item.expected_answer,
            "answer": response.answer,
            "exact": (not failed) and exact_match(
                response.answer, item.expected_answer
            ),
            "normalized": (not failed) and normalized_match(
                response.answer, item.expected_answer
            ),
            "correct": (not failed) and contains_match(
                response.answer, item.expected_answer
            ),
            "latency_sec": response.latency_sec,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,

            "cost_usd": estimate_cost(
                response.tokens_in,
                response.tokens_out,
                provider.price_in,
                provider.price_out
            ),
            "error": response.error


        })

        if delay_sec and i < len(items) -1:
            time.sleep(delay_sec)
    return results

def run_model_batch(items, provider, batch_size=10):
    """Same as run_model, but sends batch_size questions per API call.
    Latency and tokens are split equally across the questions (an estimate).
    """

    results = []

    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        n = len(chunk)

        questions = [item.question for item in chunk]

        answers, latency, t_in, t_out, error = provider.ask_batch(questions)

        if not answers:
            answers = [""] * n

        share_in = round(t_in / n)
        share_out = round(t_out / n)

        for item, answer in zip(chunk, answers):
            failed = bool(error)

            results.append({
                "model": provider.name,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "answer": answer,
                "exact": (not failed) and exact_match(
                    answer, item.expected_answer
                ),
                "normalized": (not failed) and normalized_match(
                    answer, item.expected_answer
                ),
                "correct": (not failed) and contains_match(
                    answer, item.expected_answer
                ),
                "latency_sec": round(latency / n, 2),
                "tokens_in": share_in,
                "tokens_out": share_out,
                "cost_usd": estimate_cost(
                    share_in,
                    share_out,
                    provider.price_in,
                    provider.price_out
                ),
                "error": error
            })

    return results

def run_experiment(items, providers, delay_sec=0.5,batch_size=None):

    all_results = []

    for provider in providers:
        if batch_size:
            all_results.extend(run_model_batch(items, provider, batch_size))
        else:
            results = run_model(items, provider, delay_sec)
            all_results.extend(results)

    return all_results

