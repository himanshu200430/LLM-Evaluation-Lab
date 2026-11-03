from langchain_core.embeddings import Embeddings

class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [
            [float(x) for x in vector]
            for vector in self._fn(texts)
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]
        