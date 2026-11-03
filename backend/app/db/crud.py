from datetime import datetime

from app.db.database import connect, init_db

RESULT_COLUMNS = ["model", "question", "expected_answer", "answer", "exact", "normalized",
                  "correct", "latency_sec", "tokens_in", "tokens_out", "cost_usd", "error"]

def save_experiment(dataset_name, mode, results):
    init_db()

    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO experiments(
               created_at,
               dataset_name,
               mode,
               total_questions
            )
            VALUES(?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                dataset_name,
                mode,
                len({r["question"] for r in results}),
            ),
        )

        experiment_id = cur.lastrowid
        placeholders = ", ".join("?" * (len(RESULT_COLUMNS) + 1))

        conn.executemany(
            f"""
            INSERT INTO results (
               experiment_id,
               {', '.join(RESULT_COLUMNS)})
            VALUES({placeholders})
            """,
            [
                (
                    experiment_id,
                    *[r[column] for column in RESULT_COLUMNS],

                )
                for r in results
            ],
        )
    return experiment_id

def list_experiments():

    init_db()

    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY id DESC"
        )
        return [dict(row) for row in rows]

def get_results(experiment_id):

    init_db()

    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM results WHERE experiment_id = ?",
            (experiment_id,),
        )

        return [dict(row) for row in rows]

def save_judgements(judge_model, rows):
    init_db()
    with connect() as conn:
        conn.executemany(
            "INSERT INTO judgements (result_id, judge_model, score, correct, reason, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(r["result_id"], judge_model, r["score"], r["correct"], r["reason"], r["error"])
             for r in rows],
        )

def get_judged_results(experiment_id):
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT r.*, j.score AS judge_score, j.correct AS judge_correct,
                   j.reason AS judge_reason, j.error AS judge_error
            FROM results r
            LEFT JOIN judgements j
              ON j.id = (SELECT MAX(id) FROM judgements WHERE result_id = r.id)
            WHERE r.experiment_id = ?
            """,
            (experiment_id,),
        )
        return [dict(row) for row in rows]

def save_faithfulness(judge_model, rows):
    init_db()

    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO faithfulness (
              result_id,
              judge_model,
              context,
              faithful,
              reason,
              error
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["result_id"],
                    judge_model,
                    row["context"],
                    row["faithful"],
                    row["reason"],
                    row["error"],
                )
                for row in rows
            ],
        )

def get_faithfulness(experiment_id):
    init_db()

    with connect() as conn:
        rows =  conn.execute(
            """
            SELECT
              r.id AS result_id,
              r.model,
              r.question,
              r.expected_answer,
              r.answer,
              r.correct,
              f.context,
              f.context,
              f.faithful,
              f.reason,
              f.error
            FROM faithfulness f
            JOIN results r ON r.id = f.result_id
            WHERE r.experiment_id = ?
              AND f.id = (
                  SELECT MAX(id)
                  FROM faithfulness
                  WHERE result_id = r.id
            )
            """,
            (experiment_id,),
        )
        return [dict(row) for row in rows]