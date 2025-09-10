from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from retriever.prepare_docs_strategy import PrepareDocsStrategy
from text_cleaning_strategy.base import TextCleaningStrategy


class Retriever:

    def __init__(
        self,
        db: MyWeaviateDB,
        embeddings: SentenceTransformer,
        cross_encoder: CrossEncoder,
        text_cleaning_strategy: TextCleaningStrategy
    ):
        self.db = db
        self.embeddings = embeddings
        self.cross_encoder = cross_encoder
        self.text_cleaning_strategy = text_cleaning_strategy
        
    def get_text_cleaning_strategy_name(self) -> str:
        return self.text_cleaning_strategy.get_strategy_name()

    def prepare_docs(self, prepareDocsStrategy: PrepareDocsStrategy, from_google_drive: bool = False):
        if from_google_drive:
            prepareDocsStrategy.prepare_docs_from_google_drive()
            return
        prepareDocsStrategy.prepare_docs()

    def search(self, query: str, alpha: int = 1, N: int = 20, top_k: int = 3):
        query = self.text_cleaning_strategy.clean_text(query)
        initial_retrieved_docs = self.db.search(query=query, embeddings=self.embeddings, alpha=alpha, limit=N)
        query_doc_pairs = [[query, doc.properties["content"]] for doc in initial_retrieved_docs]
        scores = self.cross_encoder.predict(query_doc_pairs)
        retrieved_docs = sorted(zip(initial_retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        return [top_k_document[0] for top_k_document in retrieved_docs[:top_k]]
