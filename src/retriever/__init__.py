from vector_db.vector_db import MyWeaviateDB
from vector_db.store_vectors import store_pdf_vectors
from sentence_transformers import SentenceTransformer
from vector_db.chunking_strategy import section_based_chunking


class Retriever:

    def __init__(
        self,
        collection_name: str,
        embeddings: SentenceTransformer,
        ef_construction: int = 300,
        bm25_b: float = 0.7,
        bm25_k1: float = 1.25
    ):
        self.db: MyWeaviateDB = MyWeaviateDB(
            ef_construction=ef_construction,
            bm25_b=bm25_b,
            bm25_k1=bm25_k1,
            collection_name=collection_name
        )
        self.embeddings = embeddings

    def prepare_docs(self, chunk: type[section_based_chunking]):
        self.db.setup_collection()
        store_pdf_vectors(db=self.db, embeddings=self.embeddings, chunk=chunk)

    def search(self, query: str, alpha: int = 1, top_k: int = 3):
        return self.db.search(query=query, embeddings=self.embeddings, alpha=alpha, k=top_k)
