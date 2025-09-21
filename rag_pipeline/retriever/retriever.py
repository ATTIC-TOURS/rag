# from ..vector_db.vector_db import MyWeaviateDB
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.cross_encoder import CrossEncoder
# from .prepare_docs_strategy import PrepareDocsStrategy
# from ..text_cleaning_strategy.base import TextCleaningStrategy
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import weaviate
from sentence_transformers import CrossEncoder
import time


connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}

def retrieve_query_relevant_docs(
    text: str, 
    similarity_top_k: int, 
    alpha: float, 
    index_name: str, 
    cross_encoder_model: str | None = None,
    rerank_top_k: int | None = None
):
    start = time.time()  # record start time
    
    client = None
    
    try:
        client = weaviate.connect_to_local(**connection_config)
        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=index_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex([], storage_context=storage_context)
        
        retriever = index.as_retriever(
                similarity_top_k=similarity_top_k,
                mode="hybrid",
                alpha=alpha
            )
        
        docs = retriever.retrieve(text)

        if cross_encoder_model:
            cross_encoder = CrossEncoder(cross_encoder_model)

            # prepare pairs: (query, doc_text)
            pairs = [(text, d.text) for d in docs]

            relevance_scores = cross_encoder.predict(pairs)

            reranked = sorted(
                zip(docs, relevance_scores),
                key=lambda x: x[1],
                reverse=True
            )

            # keep top rerank_top_k (or all if None)
            if rerank_top_k:
                reranked = reranked[:rerank_top_k]

            docs = [d for d, _ in reranked]
        end = time.time()  # record end time
    
        elapsed = end - start
        print(f"Elapsed time: {elapsed:.2f} seconds")
        
        return docs

    except Exception as e:
        print("error in retrieve_query_relevant_docs")
        print(e)
    finally:
        if client:
            client.close()


# class Retriever:

#     def __init__(
#         self,
#         db: MyWeaviateDB,
#         embeddings: SentenceTransformer,
#         cross_encoder: CrossEncoder,
#         text_cleaning_strategy: TextCleaningStrategy,
#     ):
#         self.db = db
#         self.embeddings = embeddings
#         self.cross_encoder = cross_encoder
#         self.text_cleaning_strategy = text_cleaning_strategy

#     def get_text_cleaning_strategy_name(self) -> str:
#         return self.text_cleaning_strategy.get_strategy_name()

#     def prepare_docs(
#         self, prepareDocsStrategy: PrepareDocsStrategy, from_google_drive: bool = False
#     ):
#         if from_google_drive:
#             prepareDocsStrategy.prepare_docs_from_google_drive()
#             return
#         prepareDocsStrategy.prepare_docs()

#     def search(self, query: str, alpha: float = 1, N: int = 20, top_k: int = 3):
#         query = self.text_cleaning_strategy.clean_text(query)
#         initial_retrieved_docs = self.db.search(
#             query=query, embeddings=self.embeddings, alpha=alpha, limit=N
#         )
#         if not initial_retrieved_docs:
#             return None
#         query_doc_pairs = [
#             [query, doc.properties["content"]] for doc in initial_retrieved_docs
#         ]
#         scores = self.cross_encoder.predict(query_doc_pairs)
#         retrieved_docs = sorted(
#             zip(initial_retrieved_docs, scores), key=lambda x: x[1], reverse=True
#         )
#         return [top_k_document[0] for top_k_document in retrieved_docs[:top_k]]
