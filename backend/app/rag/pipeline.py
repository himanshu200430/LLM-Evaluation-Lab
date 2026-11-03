from pathlib import Path

from app.providers.base import LLMResponse
from app.rag.chunker import split_text
from app.rag.loader import load_document
from app.rag.vector_store import VectorStore

RAG_SYSTEM = (
    "Answer the question using ONLY the context below. "
    "Reply in one short sentence or phrase. "
    "If the answer is not in the context, reply exactly: I don't know."
)

def build_index(doc_path, chunk_size=300, chunk_overlap=40, embeddings=None):
    chunks = split_text(
        load_document(doc_path),
        chunk_size,
        chunk_overlap,
    )
    store = VectorStore(embeddings)

    store.add_chunks(
        chunks,
        source=Path(doc_path).name,
    )

    return store

class RagProvider:

    def __init__(self, llm_provider, store, k=3):
        self.llm = llm_provider
        self.retriever = store.retriever(k)
        self.k = k

        self.name = f"{llm_provider.name}+rag(k={k})"
        self.model = llm_provider.model
        self.price_in = llm_provider.price_in
        self.price_out = llm_provider.price_out
        self.contexts = {}

    def build_prompt(self, question):
        documents = self.retriever.invoke(question)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )
        self.contexts[question]  = context

        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    def ask(self,question):
        return self.llm.complete(
            RAG_SYSTEM,
            self.build_prompt(question),
        )

