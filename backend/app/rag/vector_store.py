import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.rag.embeddings import ChromaDefaultEmbeddings

class VectorStore:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or ChromaDefaultEmbeddings()

        self.db = Chroma(
            collection_name = f"docs_{uuid.uuid4().hex[:8]}",
            embedding_function = self.embeddings,
        )

    def add_chunks(self, chunks, source):
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        self.db.add_documents(
            documents,
            ids=[f"{source}-{i}" for i in range(len(chunks))],
        )
    def count(self):
        return self.db._collection.count()
    
    def retriever(self, k=3):
        return self.db.as_retriever(
            search_kwargs={"k": k}
        )

    def search(self, query, k=3):
        if self.count() == 0:
            return []

        return [
            {
                "text": document.page_content,
                "source": document.metadata["source"],
                "chunk": document.metadata["chunk"],
            }

            for document in self.retriever(k).invoke(query)
        ]