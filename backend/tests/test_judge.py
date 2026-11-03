import pytest

from app.evaluation.judge import judge_answer, parse_judge_reply


class FakeLLMResponse:
    def __init__(self, answer, error=""):
        self.answer, self.error = answer, error


class FakeJudge:
    """Returns the replies you give it, one per call."""
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system_prompt, user_prompt):
        self.calls += 1
        return self.replies.pop(0)


GOOD = '{"score": 5, "correct": true, "reason": "Same meaning."}'


def test_parse_valid_reply():
    r = parse_judge_reply(GOOD)
    assert r.score == 5 and r.correct is True


def test_parse_ignores_text_and_code_fence_around_json():
    r = parse_judge_reply("Sure!\n```json\n" + GOOD + "\n```")
    assert r.score == 5


def test_parse_rejects_score_out_of_range():
    with pytest.raises(ValueError):
        parse_judge_reply('{"score": 9, "correct": true, "reason": "x"}')


def test_parse_rejects_missing_field():
    with pytest.raises(ValueError):
        parse_judge_reply('{"score": 3, "correct": false}')


def test_parse_rejects_no_json():
    with pytest.raises(ValueError):
        parse_judge_reply("I think it is correct")


def test_judge_answer_success():
    result, error = judge_answer(FakeJudge(FakeLLMResponse(GOOD)), "Q?", "Paris", "Paris")
    assert result.correct is True and error == ""


def test_judge_retries_once_on_bad_format():
    judge = FakeJudge(FakeLLMResponse("not json"), FakeLLMResponse(GOOD))
    result, error = judge_answer(judge, "Q?", "Paris", "Paris")
    assert result.score == 5 and judge.calls == 2


def test_judge_gives_up_after_two_bad_replies():
    judge = FakeJudge(FakeLLMResponse("bad"), FakeLLMResponse("worse"))
    result, error = judge_answer(judge, "Q?", "Paris", "Paris")
    assert result is None and "bad judge output" in error


def test_judge_does_not_retry_api_errors():
    judge = FakeJudge(FakeLLMResponse("", error="429 quota"))
    result, error = judge_answer(judge, "Q?", "Paris", "Paris")
    assert result is None and error == "429 quota" and judge.calls == 1