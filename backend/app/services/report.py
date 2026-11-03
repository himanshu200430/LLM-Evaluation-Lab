from app.evaluation.metrics import summarize_by_model

def model_summary(rows):
    out = []
    for model,  summary in summarize_by_model(rows).items():
        mine = [
            row
            for row in rows
            if row["model"] == model
        ]

        judged = [
            row
            for row in mine
            if row.get("judge_score") is not None
        ]

        item = dict(
            summary,
            model=model,
            judged=len(judged),
        )

        if judged:
            item["judge_accuracy_pct"] = round(sum(1 for row in judged if row["judge_correct"])/len(judged)*100,1,)
            item["avg_judge_score"] =  round(sum(row["judge_score"] for row in judged)/len(judged),2,)
        else:
            item["judge_accuracy_pct"] = None
            item["avg_judge_score"] = None
        out.append(item)
    return out

def faithfulness_summary(rows):
    models = sorted({
        row["model"]
        for row in rows
    })

    out = []

    for model in models:
        judged = [
            row
            for row in rows
            if row["model"] == model and row["faithful"] is not None
        ]

        if judged:
            faithful = sum(
                1
                for row in judged
                if row["faithful"]
            )

            out.append(
                {
                    "model":model,
                    "judged": len(judged),
                    "faithful": faithful,
                    "faithful_pct": round(
                        faithful/len(judged)*100,
                        1,
                    ),
                }
            )
    return out

def answers_table(rows):
    table = []
    for row in rows:
        judged = row.get("judge_score") is not None

        table.append(
            {
                "model": row["model"],
                "question": row["question"],
                "expected": row["expected_answer"],
                "answer": row["answer"] or "",
                "string_ok": bool(row["correct"]),
                "judge_ok": bool(row["judge_correct"]) if judged else None,
                "judge_score": row.get("judge_score"),
                "judge_reason": row.get("judge_reason") or "",
                "error": row["latency_sec"] or "",
                "latency_sec": row["latency_sec"],
                "cost_usd": row["cost_usd"],
            }
        )
    return table

def _fmt(value, digits=1):
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def experiment_markdown(
    experiment,
    rows,
    faith_rows=None,
):
    """
    README-ready markdown for one saved experiment.

    experiment = a row of crud.list_experiments()
    rows = crud.get_judged_results(id)
    """

    lines = [
        (
            f"### Experiment #{experiment['id']}: "
            f"{experiment['dataset_name']} "
            f"({experiment['mode']}, "
            f"{experiment['total_questions']} questions)"
        ),
        "",
    ]

    lines += [
        (
            "| Model | String-match acc % | Judge acc % | "
            "Avg judge score | Avg latency (s) | "
            "Cost per 1k questions ($) | Errors |"
        ),
        "|---|---|---|---|---|---|---|",
    ]

    for summary in model_summary(rows):
        per_1k = (
            summary["total_cost_usd"] / summary["total"] * 1000
            if summary["total"]
            else 0.0
        )

        lines.append(
            f"| {summary['model']} "
            f"| {_fmt(summary['accuracy_pct'])} "
            f"| {_fmt(summary['judge_accuracy_pct'])} "
            f"| {_fmt(summary['avg_judge_score'], 2)} "
            f"| {_fmt(summary['avg_latency_sec'], 2)} "
            f"| {per_1k:.4f} "
            f"| {summary['errors']} |"
        )

    judged = [
        row
        for row in answers_table(rows)
        if row["judge_ok"] is not None
    ]

    if judged:
        disagree = [
            row
            for row in judged
            if row["string_ok"] != row["judge_ok"]
        ]

        missed_by_string = sum(
            1
            for row in disagree
            if row["judge_ok"] and not row["string_ok"]
        )

        lines += [
            "",
            (
                f"String matching and the judge disagreed on "
                f"{len(disagree)} of {len(judged)} judged answers "
                f"({missed_by_string} answers that string matching "
                f"marked wrong were judged correct)."
            ),
        ]

    if faith_rows:
        lines += [
            "",
            "| Model | Faithful answers | Faithful % |",
            "|---|---|---|",
        ]

        for faith in faithfulness_summary(faith_rows):
            lines.append(
                f"| {faith['model']} "
                f"| {faith['faithful']}/{faith['judged']} "
                f"| {faith['faithful_pct']} |"
            )

    return "\n".join(lines)