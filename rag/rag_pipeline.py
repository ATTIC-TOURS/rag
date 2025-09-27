from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
import weaviate
import asyncio

connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}


async def build_two_stage_rag_pipeline(
    index_name: str,
    similarity_top_k: int,
    alpha: float,
    cross_encoder_model: str | None = None,
    rerank_top_n: int = 1,
    fact_prompt: PromptTemplate | None = None,
    answer_prompt: PromptTemplate | None = None,
):
    # 1. Connect to Weaviate (sync)
    # client = weaviate.connect_to_local(**connection_config, async_client=True)

    # Async client in v4
    client = weaviate.use_async_with_local(**connection_config)
    await client.connect()
    
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. Build index
    index = VectorStoreIndex([], storage_context=storage_context)

    # 3. Retriever
    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        mode="hybrid",
        alpha=alpha,
    )

    # 4. Optional reranker
    reranker = None
    if cross_encoder_model:
        reranker = SentenceTransformerRerank(
            model=cross_encoder_model,
            top_n=rerank_top_n,
        )

    # 5. Stage 1: Fact extraction with GPT-4o-mini
    llm_stage1 = OpenAI(model="gpt-4o-mini", request_timeout=300)
    fact_synthesizer = get_response_synthesizer(
        llm=llm_stage1,
        text_qa_template=fact_prompt,
    )

    # 6. Stage 2: Final answer formatting with GPT-5
    llm_stage2 = OpenAI(model="gpt-5", request_timeout=300)
    answer_synthesizer = get_response_synthesizer(
        llm=llm_stage2,
        text_qa_template=answer_prompt,
    )

    # 7. Custom query engine
    retriever_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker] if reranker else None,
        response_synthesizer=fact_synthesizer,
    )

    # 8. Wrap into a two-step engine
    class TwoStageQueryEngine:
        async def aquery(self, query: str):
            # Step 1: fact extraction (Response object with source_nodes)
            fact_response = await retriever_engine.aquery(query)

            # Step 2: final formatting (string only)
            final_response = await answer_synthesizer.asynthesize(
                query, fact_response.source_nodes
            )

            return {
                "facts": fact_response,  # full Response object
                "final_answer": str(final_response),  # formatted string
            }

        async def query(self, query: str):
            return await self.aquery(query)

    return TwoStageQueryEngine(), client


# def build_rag_pipeline(
#     index_name: str,
#     llm_model_name: str,
#     similarity_top_k: int,
#     alpha: float,
#     prompt_template: PromptTemplate,
#     llm_provider: LLM_PROVIDER,
#     cross_encoder_model: str | None = None,
#     rerank_top_n: int = 1,
# ):
#     """
#     Build a RAG pipeline with:
#       - Weaviate hybrid retriever
#       - Optional CrossEncoder reranker
#       - Ollama or OpenAI as generator
#     """
#     # 1. Connect to Weaviate
#     client = weaviate.connect_to_local(**connection_config)
#     vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # 2. Build index
#     index = VectorStoreIndex([], storage_context=storage_context)

#     # 3. Base retriever
#     retriever = index.as_retriever(
#         similarity_top_k=similarity_top_k,
#         mode="hybrid",
#         alpha=alpha,
#     )

#     # 4. Optional reranker
#     reranker = None
#     if cross_encoder_model:
#         reranker = SentenceTransformerRerank(
#             model=cross_encoder_model,
#             top_n=rerank_top_n,
#         )

#     # 5. LLM provider
#     if llm_provider == "ollama":
#         llm = Ollama(
#             model=llm_model_name,
#             base_url="http://localhost:11434",
#             request_timeout=300,
#         )
#     elif llm_provider == "openai":
#         llm = OpenAI(
#             model=llm_model_name,
#             request_timeout=300,
#         )
#     else:
#         raise ValueError(f"Unsupported llm provider: {llm_provider}")

#     # 7. Response synthesizer with custom prompt
#     response_synthesizer = get_response_synthesizer(
#         llm=llm,
#         text_qa_template=prompt_template,
#     )

#     # 8. Query engine
#     query_engine = RetrieverQueryEngine(
#         retriever=retriever,
#         node_postprocessors=[reranker] if reranker else None,
#         response_synthesizer=response_synthesizer,
#     )

#     return query_engine, client
