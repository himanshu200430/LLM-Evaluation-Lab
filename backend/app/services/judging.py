import time

from app.db.crud import get_results, save_judgements
from app.evaluation.judge import judge_answer

def judge_experiment(experiment_id, judge, on_progress=None, delay_sec=0.5):
    rows = [
        row
        for row in get_results(experiment_id)
        if not row["error"]
    ]

    saved = []
    failed = 0

    for i, row in enumerate(rows, start=1):
        verdict, error = judge_answer(
            judge,
            row["question"],
            row["expected_answer"],
            row["answer"],
        )

        failed += verdict is None

        saved.append(
            {
                "result_id": row["id"],
                "score": verdict.score if verdict else None,
                "correct": verdict.correct if verdict else None,
                "reason": verdict.reason if verdict else None,
                "error": error,
            }
        )
        if on_progress:
            on_progress(i, len(rows))

        if delay_sec:
            time.sleep(delay_sec)

    save_judgements(judge.name, saved)
    return len(rows) - failed, failed
