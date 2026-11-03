import pytest

from app.evaluation.exact_match import contains_match, exact_match, normalize, normalized_match
from app.evaluation.metrics import summarize
from app.providers.base import BaseProvider, LLMResponse
from app.schemas.dataset import QAItem
from app.services.cost_calculator import estimate_cost
from app.services.experiment_runner import run_model


class FakeProvider(BaseProvider):
    name = "fake"
    model = "fake-1"
    price_in = 1.0
    price_out = 2.0

    def ask(self, question):
        return LLMResponse(answer="The capital is Paris.", latency_sec=0.5, tokens_in=10, tokens_out=5)


def test_normalize():
    assert normalize("  Paris!!  ") == "paris"
    assert normalize("The  Capital, is Paris.") == "the capital is paris"


def test_exact_match_is_strict():
    assert exact_match("Paris", "Paris")
    assert not exact_match("paris", "Paris")


def test_normalized_match_ignores_case_and_punctuation():
    assert normalized_match("Paris.", "paris")
    assert not normalized_match("The capital is Paris", "Paris")


def test_contains_match():
    assert contains_match("The capital is Paris.", "Paris")
    assert not contains_match("The answer is 196", "96")
    assert not contains_match("Lyon", "Paris")


def test_estimate_cost():
    assert estimate_cost(1000, 500, 0.10, 0.40) == pytest.approx(0.0003)


def test_summarize():
    results = [
        {"correct": True, "normalized": True, "latency_sec": 1.0,
         "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.001, "error": ""},
        {"correct": False, "normalized": False, "latency_sec": 3.0,
         "tokens_in": 20, "tokens_out": 5, "cost_usd": 0.002, "error": "timeout"},
    ]
    s = summarize(results)
    assert s["total"] == 2
    assert s["accuracy_pct"] == 50.0
    assert s["errors"] == 1
    assert s["avg_latency_sec"] == 2.0
    assert s["tokens_in"] == 30
    assert s["total_cost_usd"] == pytest.approx(0.003)


def test_summarize_empty():
    s = summarize([])
    assert s["total"] == 0
    assert s["accuracy_pct"] == 0.0


def test_run_model_with_fake_provider():
    items = [QAItem(question="Capital of France?", expected_answer="Paris")]
    r = run_model(items, FakeProvider(), delay_sec=0)[0]
    assert r["correct"] is True        # answer contains "Paris"
    assert r["normalized"] is False    # but it is not only "Paris"
    assert r["cost_usd"] == pytest.approx(0.00002)
    assert r["error"] == ""