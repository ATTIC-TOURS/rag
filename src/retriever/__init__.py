from vector_db.vector_db import MyWeaviateDB
from vector_db.store_vectors import store_pdf_vectors
from sentence_transformers import SentenceTransformer
from vector_db.chunking_strategy import section_based_chunking


class Retriever:

    def __init__(
        self,
        embeddings: SentenceTransformer,
        ef_construction: int = 300,
        bm25_b: float = 0.7,
        bm25_k1: float = 1.25,
    ):
        self.db: MyWeaviateDB = MyWeaviateDB(
            embeddings=embeddings,
            ef_construction=ef_construction,
            bm25_b=bm25_b,
            bm25_k1=bm25_k1,
        )

    def pre_compute_docs(self, chunk: type[section_based_chunking]):
        self.db.setup_collection()
        store_pdf_vectors(self.db, chunk)

    def retrieve_relevant_docs(self, query: str, alpha: int = 1, k: int = 3):
        return self.db.search(query=query, alpha=alpha, k=k)
