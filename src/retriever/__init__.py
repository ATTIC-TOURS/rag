from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from retriever.prepare_docs_strategy import PrepareDocsStrategy
from text_cleaning_strategy.base import TextCleaningStrategy


class Retriever:

    def __init__(
        self,
        db: MyWeaviateDB,
        embeddings: SentenceTransformer,
        text_cleaning_strategy: TextCleaningStrategy
    ):
        self.db = db
        self.embeddings = embeddings
        self.text_cleaning_strategy = text_cleaning_strategy
        
    def get_text_cleaning_strategy_name(self) -> str:
        return self.text_cleaning_strategy.get_strategy_name()

    def prepare_docs(self, prepareDocsStrategy: PrepareDocsStrategy):
        prepareDocsStrategy.prepare_docs()

    def search(self, query: str, alpha: int = 1, top_k: int = 3):
        query = self.text_cleaning_strategy.clean_text(query)
        return self.db.search(query=query, embeddings=self.embeddings, alpha=alpha, top_k=top_k)
