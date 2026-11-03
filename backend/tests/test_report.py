import pytest

from app.db import crud, database
from app.services.judging import judge_experiment
from app.services.report import answers_table, faithfulness_summary, model_summary


def make_result(model, question, correct, error=""):
    return {"model": model, "question": question, "expected_answer": "A", "answer": "" if error else "A",
            "exact": correct, "normalized": correct, "correct": correct,
            "latency_sec": 1.0, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.001, "error": error}


class FakeLLMResponse:
    def __init__(self, answer, error=""):
        self.answer, self.error = answer, error


class FakeJudge:
    name = "fake-judge"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def complete(self, system_prompt, user_prompt):
        return FakeLLMResponse(self.verdicts.pop(0))


def seed(tmp_path, monkeypatch, results):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    return crud.save_experiment("a.csv", "per-question", results)


def test_model_summary_without_judging(tmp_path, monkeypatch):
    exp = seed(tmp_path, monkeypatch, [make_result("m1", "Q1?", True), make_result("m1", "Q2?", False)])
    summary = model_summary(crud.get_judged_results(exp))[0]
    assert summary["model"] == "m1" and summary["accuracy_pct"] == 50.0
    assert summary["judge_accuracy_pct"] is None and summary["judged"] == 0


def test_model_summary_with_judging(tmp_path, monkeypatch):
    exp = seed(tmp_path, monkeypatch, [make_result("m1", "Q1?", False), make_result("m1", "Q2?", False)])
    good = '{"score": 5, "correct": true, "reason": "same meaning"}'
    bad = '{"score": 1, "correct": false, "reason": "wrong"}'
    ok, failed = judge_experiment(exp, FakeJudge([good, bad]), delay_sec=0)
    assert (ok, failed) == (2, 0)

    summary = model_summary(crud.get_judged_results(exp))[0]
    assert summary["accuracy_pct"] == 0.0            # string matching said both wrong
    assert summary["judge_accuracy_pct"] == 50.0     # judge disagreed on one of them
    assert summary["avg_judge_score"] == 3.0


def test_judge_skips_failed_answers_and_counts_judge_failures(tmp_path, monkeypatch):
    exp = seed(tmp_path, monkeypatch, [make_result("m1", "Q1?", True),
                                       make_result("m1", "Q2?", False, error="429"),
                                       make_result("m1", "Q3?", True)])
    good = '{"score": 5, "correct": true, "reason": "ok"}'
    ok, failed = judge_experiment(exp, FakeJudge([good, "not json", "still not json"]), delay_sec=0)
    assert ok == 1 and failed == 1                   # Q2 had an API error, so only 2 were judged


def test_answers_table_shows_both_verdicts(tmp_path, monkeypatch):
    exp = seed(tmp_path, monkeypatch, [make_result("m1", "Q1?", False)])
    judge_experiment(exp, FakeJudge(['{"score": 4, "correct": true, "reason": "fine"}']), delay_sec=0)
    row = answers_table(crud.get_judged_results(exp))[0]
    assert row["string_ok"] is False and row["judge_ok"] is True and row["judge_score"] == 4


def test_answers_table_unjudged_has_no_judge_verdict(tmp_path, monkeypatch):
    exp = seed(tmp_path, monkeypatch, [make_result("m1", "Q1?", True)])
    row = answers_table(crud.get_judged_results(exp))[0]
    assert row["judge_ok"] is None and row["judge_score"] is None


def test_faithfulness_summary(tmp_path, monkeypatch):
    exp = seed(tmp_path, monkeypatch, [make_result("m1+rag", "Q1?", True), make_result("m1+rag", "Q2?", True)])
    ids = {r["question"]: r["id"] for r in crud.get_results(exp)}
    crud.save_faithfulness("j", [
        {"result_id": ids["Q1?"], "context": "c", "faithful": True, "reason": "ok", "error": ""},
        {"result_id": ids["Q2?"], "context": "c", "faithful": False, "reason": "invented", "error": ""},
    ])
    summary = faithfulness_summary(crud.get_faithfulness(exp))
    assert summary == [{"model": "m1+rag", "judged": 2, "faithful": 1, "faithful_pct": 50.0}]