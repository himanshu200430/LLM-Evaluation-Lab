from pathlib import Path

def load_document(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8-sig")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    raise ValueError(
        "Unsupported document type. Use .txt, .md or .pdf"
    )