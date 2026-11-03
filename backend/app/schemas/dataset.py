"""Pydantic schema: what one row of a test dataset must look like."""
from typing import Optional
from pydantic import BaseModel, field_validator


class QAItem(BaseModel):
    question: str
    expected_answer: str
    context: Optional[str] = None   
    source_doc: Optional[str] = None   

    @field_validator("question", "expected_answer")
    @classmethod
    def check_empty(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Field must not be empty")
        return value.strip()