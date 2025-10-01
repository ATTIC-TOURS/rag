from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
import weaviate, hashlib

connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}


# --- Query expansion helper ---
async def expand_query(query: str, num_expansions: int = 3) -> list[str]:
    llm = OpenAI(model="gpt-5-mini", request_timeout=60)
    prompt = (
        f"Generate {num_expansions} alternative queries that mean the same as:\n"
        f"'{query}'\n"
        "Return each variation as a short standalone query."
    )
    response = await llm.acomplete(prompt)
    expansions = [q.strip("-• \n") for q in response.text.split("\n") if q.strip()]
    return [query] + expansions  # include original


# --- Deduplication helper ---
def deduplicate_nodes(nodes):
    seen = set()
    unique_nodes = []
    for node in nodes:
        node_hash = (
            getattr(node, "node_id", None)
            or hashlib.md5(node.text.encode("utf-8")).hexdigest()
        )
        if node_hash not in seen:
            seen.add(node_hash)
            unique_nodes.append(node)
    return unique_nodes


# --- Build pipeline ---
async def build_rag_pipeline(
    index_name: str,
    alpha: float,
    cross_encoder_model: str | None = None,
    rerank_top_n: int = 5,  # final cut
    fact_prompt: PromptTemplate | None = None,
    answer_prompt: PromptTemplate | None = None,
    two_stage: bool = True,
    use_query_expansion: bool = False,
    query_expansion_num: int = 3,
    base_k: int = 10,  # docs for original query
    expansion_k: int = 5,  # docs for each expansion
):
    # 1. Connect to Weaviate
    client = weaviate.use_async_with_local(
        port=8080, grpc_port=50051, skip_init_checks=True
    )
    await client.connect()

    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. Build index + retriever
    index = VectorStoreIndex([], storage_context=storage_context)
    base_retriever = index.as_retriever(
        similarity_top_k=base_k, mode="hybrid", alpha=alpha
    )
    expansion_retriever = index.as_retriever(
        similarity_top_k=expansion_k, mode="hybrid", alpha=alpha
    )

    # 3. Optional reranker
    reranker = None
    if cross_encoder_model:
        reranker = SentenceTransformerRerank(
            model=cross_encoder_model, top_n=rerank_top_n
        )

    # 4. Synthesizers
    llm_stage1 = OpenAI(model="gpt-5-mini", request_timeout=300)
    fact_synthesizer = get_response_synthesizer(
        llm=llm_stage1, text_qa_template=fact_prompt
    )

    answer_synthesizer = None
    if two_stage:
        llm_stage2 = OpenAI(model="gpt-5", request_timeout=300)
        answer_synthesizer = get_response_synthesizer(
            llm=llm_stage2, text_qa_template=answer_prompt
        )

    # --- Query Engine Wrapper ---
    class FlexibleQueryEngine:
        async def aquery(self, query: str):
            # Step 0: query expansion
            if use_query_expansion:
                expanded_queries = await expand_query(query, query_expansion_num)
            else:
                expanded_queries = [query]

            # Step 1: retrieve docs per query
            all_nodes = []
            for i, q in enumerate(expanded_queries):
                if i == 0:
                    retrieved = await base_retriever.aretrieve(q)
                else:
                    retrieved = await expansion_retriever.aretrieve(q)
                all_nodes.extend(retrieved)

            # Step 2: deduplicate
            unique_nodes = deduplicate_nodes(all_nodes)

            # Step 3: rerank globally (if enabled)
            if reranker:
                unique_nodes = reranker.postprocess_nodes(
                    query_str=query, nodes=unique_nodes
                )

            # Keep only top N
            final_nodes = unique_nodes[:rerank_top_n]

            # Step 4: fact synthesis
            fact_response = await fact_synthesizer.asynthesize(query, final_nodes)

            # Step 5: final synthesis (optional)
            if two_stage and answer_synthesizer:
                final_response = await answer_synthesizer.asynthesize(
                    query, fact_response.source_nodes
                )
                return {
                    "expanded_queries": expanded_queries,
                    "retrieved_docs": len(all_nodes),
                    "unique_docs": len(unique_nodes),
                    "final_answer": str(final_response),
                    "facts": fact_response,
                }
            else:
                return {
                    "expanded_queries": expanded_queries,
                    "retrieved_docs": len(all_nodes),
                    "unique_docs": len(unique_nodes),
                    "final_answer": str(fact_response),
                    "facts": fact_response,
                }

        async def query(self, query: str):
            return await self.aquery(query)

    return FlexibleQueryEngine(), client


# def build_rag_pipeline(
#     index_name: str,
#     llm_model_name: str,
#     similarity_top_k: int,
#     alpha: float,
#     prompt_template: PromptTemplate,
#     llm_provider: str,
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
