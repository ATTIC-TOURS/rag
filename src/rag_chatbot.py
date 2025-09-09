from retriever import Retriever
from colorama import init
from sentence_transformers import SentenceTransformer
import ollama
from retriever.prepare_docs_strategy import PrepareDocsStrategy
from text_cleaning_strategy.base import TextCleaningStrategy
from text_cleaning_strategy.docs.v1 import DocsCleaningStrategyV1
from text_cleaning_strategy.docs.v2 import DocsCleaningStrategyV2
from text_cleaning_strategy.query.v1 import QueryCleaningStrategyV1
from chunking_strategy.base import ChunkingStrategy
from chunking_strategy.v1 import ChunkingStrategyV1
from chunking_strategy.fixed_window_chunking import FixedWindowChunking
from vector_db.vector_db import MyWeaviateDB
from prompts.strategy_base import PromptStrategy
from prompts.strategy_v1 import PromptStrategyV1
from sentence_transformers.cross_encoder import CrossEncoder
from contextual_retrieval.context_embedder import ContextEmbedderLLM
from chunking_strategy.pdf_based_recursively_split_chunking import (
    PdfBasedRecursivelySplitChunking,
)

init(autoreset=True)


class RAG_Chatbot:

    def __init__(self):
        self.embeddings: SentenceTransformer = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )
        self.db: MyWeaviateDB = MyWeaviateDB(
            ef_construction=300, bm25_b=0.7, bm25_k1=1.25
        )
        query_cleaning_strategy = QueryCleaningStrategyV1()
        cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.retriever = Retriever(
            db=self.db,
            embeddings=self.embeddings,
            text_cleaning_strategy=query_cleaning_strategy,
            cross_encoder=cross_encoder,
        )

    def prepare_docs(self) -> None:

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
        self.retriever.prepare_docs(prepareDocsStrategy=prepareDocsStrategy)

    def _retrieved_relevant_docs(
        self, query: str, alpha: int = 0.8, top_k: int = 3
    ) -> list[str]:
        relevant_docs = []
        for relevant_doc in self.retriever.search(query, alpha=alpha, top_k=top_k):
            relevant_docs.append(relevant_doc.properties["content"])
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

    def answer(self, query: str) -> str:
        relevant_docs = self._retrieved_relevant_docs(query, top_k=5)  # retriever
        messages = self._get_messages(query=query, context=relevant_docs)  # prompt
        return self._generate_response(messages)  # generation


def main():
    chatbot = RAG_Chatbot()
    while True:
        query = input('query:')
        relevant_docs = chatbot._retrieved_relevant_docs(query)
        messages = chatbot._get_messages(query, relevant_docs)
        print(f'{messages[0]["content"]} {messages[1]["content"]}')


if __name__ == "__main__":
    main()
