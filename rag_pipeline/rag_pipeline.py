from .retriever.retriever import Retriever
from colorama import init, Fore
from sentence_transformers import SentenceTransformer
import ollama
from .retriever.prepare_docs_strategy import PrepareDocsStrategy
from .text_cleaning_strategy.base import TextCleaningStrategy
from .text_cleaning_strategy.docs.v1 import DocsCleaningStrategyV1
from .text_cleaning_strategy.docs.v2 import DocsCleaningStrategyV2
from .text_cleaning_strategy.query.v1 import QueryCleaningStrategyV1
from .chunking_strategy.base import ChunkingStrategy
from .chunking_strategy.v1 import ChunkingStrategyV1
from .chunking_strategy.fixed_window_chunking import FixedWindowChunking
from .vector_db.vector_db import MyWeaviateDB
from .prompts.strategy_base import PromptStrategy
from .prompts.strategy_v1 import PromptStrategyV1
from sentence_transformers.cross_encoder import CrossEncoder
from .contextual_retrieval.context_embedder import ContextEmbedderLLM
from .chunking_strategy.pdf_based_recursively_split_chunking import (
    PdfBasedRecursivelySplitChunking,
)
from sklearn.pipeline import Pipeline
import joblib
from .classifier.japan_visa_related_or_not.modules import (
    MyTextCleaner,
    MyEmbeddingTransformer,
)

init(autoreset=True)


class RagPipeline:

    def __init__(self, collection_name: str = "Requirements"):
        classifier = joblib.load(
            "rag_pipeline/classifier/japan_visa_related_or_not/japan_visa_related_classifier.pkl"
        )
        self.related_or_not_clf = Pipeline(
            [
                ("text_cleaner", MyTextCleaner()),
                (
                    "embedder",
                    MyEmbeddingTransformer(model_name="intfloat/multilingual-e5-base"),
                ),
                ("classifier", classifier),
            ]
        )
        self.embeddings: SentenceTransformer = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )
        self.db: MyWeaviateDB = MyWeaviateDB(
            ef_construction=300,
            bm25_b=0.7,
            bm25_k1=1.25,
            collection_name=collection_name,
        )
        query_cleaning_strategy = QueryCleaningStrategyV1()
        cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.retriever = Retriever(
            db=self.db,
            embeddings=self.embeddings,
            text_cleaning_strategy=query_cleaning_strategy,
            cross_encoder=cross_encoder,
        )

    def prepare_docs(self, from_google_drive: bool = False) -> None:

        docs_cleaning_strategy: TextCleaningStrategy = DocsCleaningStrategyV2()

        max_tokens = self.embeddings.get_max_seq_length()
        chunking_strategy = PdfBasedRecursivelySplitChunking(
            chunk_overlap_rate=0.2, max_tokens=max_tokens
        )

        context_embedder = ContextEmbedderLLM(model_name="gemma3:1b")

        prepareDocsStrategy = PrepareDocsStrategy(
            db=self.db,
            embeddings=self.embeddings,
            text_cleaning_strategy=docs_cleaning_strategy,
            chunking_strategy=chunking_strategy,
            context_embedder=context_embedder,
        )
        self.retriever.prepare_docs(
            prepareDocsStrategy=prepareDocsStrategy, from_google_drive=from_google_drive
        )

    def _retrieved_relevant_docs(
        self, query: str, alpha: float = 0.8, top_k: int = 3
    ) -> list[str]:
        relevant_docs = []
        query_relevant_docs = self.retriever.search(query, alpha=alpha, top_k=top_k)
        if not query_relevant_docs:
            return []
        for query_relevant_doc in query_relevant_docs:
            relevant_docs.append(query_relevant_doc.properties["content"])
        return relevant_docs

    def _get_messages(self, query: str, context: list[str]) -> list[dict[str, str]]:
        promptStrategy: PromptStrategy = PromptStrategyV1()
        return promptStrategy.get_messages(query, context)

    def _generate_response(self, messages: list[dict[str, str]]):
        stream = ollama.chat(model="gemma3:1b", messages=messages, stream=True)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def answer(self, query: str):
        is_japan_visa_related = bool(self.related_or_not_clf.predict([query]))
        if is_japan_visa_related:
            relevant_docs = self._retrieved_relevant_docs(query, top_k=5)  # retriever
            messages = self._get_messages(query=query, context=relevant_docs)  # prompt
            return self._generate_response(messages)  # generation
        else:
            return self._generate_response(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are assistant in Japan Visa related in Attic Tours Company."
                            "You should be helpful."
                            "The response must be restrictly related in Japan Visa."
                            "You have to respect them."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"respond to the following query\nquery: {query}",
                    },
                ]
            )
