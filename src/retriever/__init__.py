from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from retriever.prepare_docs_strategy import PrepareDocsStrategy


class Retriever:

    def __init__(
        self,
        db: MyWeaviateDB,
        embeddings: SentenceTransformer,
    ):
        self.db = db
        self.embeddings = embeddings

    def prepare_docs(self, prepareDocsStrategy: PrepareDocsStrategy):
        prepareDocsStrategy.prepare_docs()

    def search(self, query: str, alpha: int = 1, top_k: int = 3):
        return self.db.search(query=query, embeddings=self.embeddings, alpha=alpha, top_k=top_k)
