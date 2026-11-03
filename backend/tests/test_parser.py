import pytest

from app.services.dataset_parser import DatasetError, parse_dataset

GOOD_CSV = b"question,expected_answer\nWhat is 2+2?,4\nCapital of France?,Paris\n"


def test_valid_csv():
    items = parse_dataset("test.csv", GOOD_CSV)
    assert len(items) == 2
    assert items[0].question == "What is 2+2?"
    assert items[0].context is None  # optional field defaults to None


def test_csv_with_optional_columns():
    data = b"question,expected_answer,context,source_doc\nQ?,A,some context,doc.pdf\n"
    item = parse_dataset("t.csv", data)[0]
    assert item.context == "some context"
    assert item.source_doc == "doc.pdf"


def test_empty_optional_becomes_none():
    data = b"question,expected_answer,context\nQ?,A,\n"
    assert parse_dataset("t.csv", data)[0].context is None


def test_column_names_are_case_insensitive():
    data = b"Question,Expected_Answer\nQ?,A\n"
    assert len(parse_dataset("t.csv", data)) == 1


def test_excel_bom_is_handled():
    data = b"\xef\xbb\xbf" + GOOD_CSV
    assert len(parse_dataset("t.csv", data)) == 2


def test_missing_required_column():
    with pytest.raises(DatasetError, match="expected_answer"):
        parse_dataset("t.csv", b"question,answer\nQ?,A\n")


def test_empty_question_reports_row_number():
    data = b"question,expected_answer\nGood?,Yes\n,No answer here\n"
    with pytest.raises(DatasetError, match="Row 3"):
        parse_dataset("t.csv", data)


def test_header_only_file_is_rejected():
    with pytest.raises(DatasetError, match="no rows"):
        parse_dataset("t.csv", b"question,expected_answer\n")


def test_valid_jsonl():
    data = b'{"question": "Q1?", "expected_answer": "A1"}\n\n{"question": "Q2?", "expected_answer": "A2", "context": "c"}\n'
    items = parse_dataset("t.jsonl", data)
    assert len(items) == 2
    assert items[1].context == "c"


def test_bad_jsonl_line():
    data = b'{"question": "Q1?", "expected_answer": "A1"}\nnot json\n'
    with pytest.raises(DatasetError, match="Line 2"):
        parse_dataset("t.jsonl", data)


def test_unsupported_extension():
    with pytest.raises(DatasetError, match="Unsupported"):
        parse_dataset("t.xlsx", GOOD_CSV)


def test_non_utf8_file():
    with pytest.raises(DatasetError, match="UTF-8"):
        parse_dataset("t.csv", b"\xff\xfe\x00bad")


def test_sample_dataset_is_valid():
    with open("../data/sample_questions.csv", "rb") as f:
        items = parse_dataset("sample_questions.csv", f.read())
    assert len(items) == 10