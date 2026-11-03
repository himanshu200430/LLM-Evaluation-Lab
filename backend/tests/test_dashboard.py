"""Opens the Streamlit app headlessly and checks every page renders without an error."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.db import crud, database

DASHBOARD = str(Path(__file__).resolve().parents[1] / "dashboard.py")


def make_result(model, question, correct):
    return {"model": model, "question": question, "expected_answer": "A", "answer": "A",
            "exact": correct, "normalized": correct, "correct": correct,
            "latency_sec": 1.0, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.001, "error": ""}


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    exp = crud.save_experiment("a.csv", "per-question", [
        make_result("m1", "Q1?", True), make_result("m1", "Q2?", False),
        make_result("m2", "Q1?", True), make_result("m2", "Q2?", True),
    ])
    ids = {(r["model"], r["question"]): r["id"] for r in crud.get_results(exp)}
    crud.save_judgements("judge", [
        {"result_id": ids[("m1", "Q2?")], "score": 5, "correct": True, "reason": "same meaning", "error": ""},
        {"result_id": ids[("m2", "Q1?")], "score": 2, "correct": False, "reason": "wrong", "error": ""},
    ])
    crud.save_faithfulness("judge", [
        {"result_id": ids[("m1", "Q1?")], "context": "ctx", "faithful": False, "reason": "invented", "error": ""},
    ])
    return exp


def test_run_page_loads():
    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Run experiment"


def test_results_page_with_no_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "empty.db")
    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert any("No experiments yet" in i.value for i in app.info)


def test_results_page_shows_data_and_filters_work(seeded_db):
    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert len(app.dataframe) >= 2                       # summary table + answers table

    app.checkbox[0].check().run()                        # only wrong answers
    assert not app.exception
    app.checkbox[1].check().run()                        # only disagreements
    assert not app.exception