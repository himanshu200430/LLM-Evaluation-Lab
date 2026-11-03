import pytest

from app.db import crud, database
from app.evaluation.faithfulness import judge_faithfulness, parse_faithfulness_reply


class FakeLLMResponse:
    def __init__(self, answer, error=""):
        self.answer, self.error = answer, error


class FakeJudge:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_prompt = None

    def complete(self, system_prompt, user_prompt):
        self.calls += 1
        self.last_prompt = user_prompt
        return self.replies.pop(0)


GOOD = '{"faithful": true, "reason": "The context says 4 months."}'


def test_parse_valid_reply():
    r = parse_faithfulness_reply(GOOD)
    assert r.faithful is True and "4 months" in r.reason


def test_parse_ignores_text_around_json():
    assert parse_faithfulness_reply("Here you go:\n```json\n" + GOOD + "\n```").faithful is True


def test_parse_rejects_missing_field():
    with pytest.raises(ValueError):
        parse_faithfulness_reply('{"faithful": true}')


def test_parse_rejects_no_json():
    with pytest.raises(ValueError):
        parse_faithfulness_reply("looks fine to me")


def test_judge_sends_question_context_and_answer():
    judge = FakeJudge(FakeLLMResponse(GOOD))
    result, error = judge_faithfulness(judge, "How long?", "The probation period is 4 months.", "4 months")
    assert result.faithful is True and error == ""
    assert "How long?" in judge.last_prompt
    assert "The probation period is 4 months." in judge.last_prompt
    assert "Answer: 4 months" in judge.last_prompt


def test_judge_retries_once_on_bad_format():
    judge = FakeJudge(FakeLLMResponse("not json"), FakeLLMResponse(GOOD))
    result, error = judge_faithfulness(judge, "Q?", "ctx", "a")
    assert result.faithful is True and judge.calls == 2


def test_judge_does_not_retry_api_errors():
    judge = FakeJudge(FakeLLMResponse("", error="429 quota"))
    result, error = judge_faithfulness(judge, "Q?", "ctx", "a")
    assert result is None and error == "429 quota" and judge.calls == 1


def make_result(question):
    return {"model": "m1+rag(k=3)", "question": question, "expected_answer": "A", "answer": "A",
            "exact": True, "normalized": True, "correct": True,
            "latency_sec": 0.5, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.001, "error": ""}


def test_faithfulness_saved_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    exp_id = crud.save_experiment("a.csv", "rag", [make_result("Q1?"), make_result("Q2?")])
    ids = {r["question"]: r["id"] for r in crud.get_results(exp_id)}

    crud.save_faithfulness("judge", [
        {"result_id": ids["Q1?"], "context": "ctx one", "faithful": True, "reason": "ok", "error": ""},
        {"result_id": ids["Q2?"], "context": "ctx two", "faithful": False, "reason": "invented", "error": ""},
    ])
    rows = {r["question"]: r for r in crud.get_faithfulness(exp_id)}
    assert rows["Q1?"]["faithful"] == 1 and rows["Q1?"]["context"] == "ctx one"
    assert rows["Q2?"]["faithful"] == 0 and rows["Q2?"]["reason"] == "invented"


def test_rejudging_keeps_only_latest_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    exp_id = crud.save_experiment("a.csv", "rag", [make_result("Q1?")])
    rid = crud.get_results(exp_id)[0]["id"]
    crud.save_faithfulness("j", [{"result_id": rid, "context": "c", "faithful": False, "reason": "old", "error": ""}])
    crud.save_faithfulness("j", [{"result_id": rid, "context": "c", "faithful": True, "reason": "new", "error": ""}])
    rows = crud.get_faithfulness(exp_id)
    assert len(rows) == 1 and rows[0]["reason"] == "new"