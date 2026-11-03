from app.db import crud, database
from app.evaluation.metrics import summarize_by_model


def make_result(model, question, correct):
    return {"model": model, "question": question, "expected_answer": "A", "answer": "A",
            "exact": correct, "normalized": correct, "correct": correct,
            "latency_sec": 0.5, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.001, "error": ""}


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    results = [make_result("m1", "Q1?", True), make_result("m2", "Q1?", False)]

    exp_id = crud.save_experiment("sample.csv", "batch(10)", results)

    assert crud.list_experiments()[0]["dataset_name"] == "sample.csv"
    rows = crud.get_results(exp_id)
    assert len(rows) == 2
    summary = summarize_by_model(rows)
    assert summary["m1"]["accuracy_pct"] == 100.0
    assert summary["m2"]["accuracy_pct"] == 0.0


def test_experiments_are_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    first = crud.save_experiment("a.csv", "per-question", [make_result("m1", "Q?", True)])
    second = crud.save_experiment("b.csv", "per-question", [make_result("m1", "Q?", True)])
    assert first != second
    assert len(crud.get_results(first)) == 1
    assert len(crud.list_experiments()) == 2

def test_judgements_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    exp_id = crud.save_experiment("a.csv", "batch(10)", [make_result("m1", "Q1?", False), make_result("m1", "Q2?", True)])
    ids = [r["id"] for r in crud.get_results(exp_id)]

    assert all(r["judge_score"] is None for r in crud.get_judged_results(exp_id))

    crud.save_judgements("judge-model", [
        {"result_id": ids[0], "score": 5, "correct": True, "reason": "same meaning", "error": ""},
        {"result_id": ids[1], "score": None, "correct": None, "reason": None, "error": "429"},
    ])
    judged = {r["id"]: r for r in crud.get_judged_results(exp_id)}
    assert judged[ids[0]]["judge_score"] == 5 and judged[ids[0]]["judge_correct"] == 1
    assert judged[ids[1]]["judge_score"] is None and judged[ids[1]]["judge_error"] == "429"