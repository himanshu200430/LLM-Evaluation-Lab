"""FastAPI entry point. Run with:  uvicorn app.main:app --reload"""
from app.config import MAX_UPLOAD_MB
from app.services.dataset_parser import DatasetError, parse_dataset
from fastapi import FastAPI, File, HTTPException, UploadFile
app = FastAPI(title="LLM EvalLab")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/datasets/validate")
async def validate_dataset(file: UploadFile = File(...)):
    """Upload a dataset and check that it is valid (nothing is saved yet)."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB")
    
    try:
        items = parse_dataset(file.filename, content)
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": file.filename,
        "total_questions": len(items),
        "with_context": sum(1 for i in items if i.context),
        "preview": [i.model_dump() for i in items[:3]],
    }