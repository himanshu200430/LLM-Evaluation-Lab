"""Read a CSV or JSONL file and turn it into a validated list of QAItem."""
import csv
import io
import json

from pydantic import ValidationError
from app.schemas.dataset import QAItem

REQUIRED_COLUMNS = {"question", "expected_answer"}
ALLOWED_COLUMNS = set(QAItem.model_fields)
MAX_ERRORS_SHOWN = 5

class DatasetError(ValueError):
    pass


def _clean_row(row):
    cleaned ={}
    for key, value in row.items():
        if key is None:
            continue
        key = key.strip().lower()

        if key not in ALLOWED_COLUMNS:
            continue

        if isinstance(value,str):
            value = value.strip()

        if value == "":
            value = None

        cleaned[key] = value
    return cleaned

def build_items(rows):
    items = []
    errors = []

    for number, row in rows:
        try:
            item = QAItem(**_clean_row(row))
            items.append(item)

        except ValidationError as e:
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "row"
                errors.append(
                    f"Row {number}: {field} - {error['msg']}"
                )
    if errors:
        message = "; ".join(errors[:MAX_ERRORS_SHOWN])
        if len(errors) > MAX_ERRORS_SHOWN:
            extra = len(errors) - MAX_ERRORS_SHOWN
            message += f" (and {extra} more)"

        raise DatasetError(message)

    if not items:
        raise DatasetError("The dataset has no rows.")
    return items

def parse_csv(text):
    reader = csv.DictReader(io.StringIO(text))

    columns = set()

    for column in reader.fieldnames or []:
        columns.add(column.strip().lower())

    missing = REQUIRED_COLUMNS - columns

    if missing:
        raise DatasetError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )    
    rows = []

    for number, row in enumerate(reader, start=2):
        rows.append((number,row))

    return build_items(rows)

def parse_jsonl(text):
    rows = []

    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            row = json.loads(line)

        except json.JSONDecodeError:
            raise DatasetError(f"Line {number}: invalid JSON")

        if not isinstance(row, dict):
            raise DatasetError(
                f"Line {number}: each line must be a JSON object"
            )
        rows.append((number,row))

    return build_items(rows)

def parse_dataset(filename,content):
    try:
        text = content.decode("utf-8-sig")

    except UnicodeDecodeError:
        raise DatasetError("File must use UTF-8 encoding.")
    filename = (filename or "").lower()

    if filename.endswith(".csv"):
        return parse_csv(text)
    
    if filename.endswith(".jsonl"):
        return parse_jsonl(text)
    raise DatasetError("Unsupported file type. Upload a .csv or .jsonl file.")