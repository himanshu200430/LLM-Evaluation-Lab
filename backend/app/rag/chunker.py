from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=300, chunk_overlap=40):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )

    return [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if chunk.strip()
    ]