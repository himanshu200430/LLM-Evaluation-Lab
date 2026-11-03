from pydantic import BaseModel, Field

class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    correct: bool
    reason: str

class FaithfulnessResult(BaseModel):
    faithful: bool
    reason: str   