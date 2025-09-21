from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI
import weaviate
from .preprocessing.preprocessing import set_embeddings

connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}

def build_rag_pipeline(
    index_name: str,
    embeddings_model_name: str,
    llm_model_name: str,
    similarity_top_k: int,
    alpha: float,
    prompt_template: PromptTemplate,
    cross_encoder_model: str | None = None,
    rerank_top_n: int | None = None,
    llm_provider: str = "ollama",  # "ollama" or "openai"
    openai_api_key: str | None = None,
):
    """
    Build a RAG pipeline with:
      - Weaviate hybrid retriever
      - Optional CrossEncoder reranker
      - Ollama or OpenAI as generator
    """
    # 1. Connect to Weaviate
    client = weaviate.connect_to_local(**connection_config)
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2. Set embeddings
    set_embeddings(embeddings_model_name)

    # 3. Build index
    index = VectorStoreIndex([], storage_context=storage_context)

    # 4. Base retriever
    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        mode="hybrid",
        alpha=alpha,
    )

    # 5. Optional reranker
    reranker = None
    if cross_encoder_model:
        reranker = SentenceTransformerRerank(
            model=cross_encoder_model,
            top_n=rerank_top_n,
        )

    # 6. LLM provider
    if llm_provider == "ollama":
        llm = Ollama(
            model=llm_model_name,
            base_url="http://localhost:11434",
            request_timeout=300,
        )
    elif llm_provider == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI provider selected but no API key provided.")
        llm = OpenAI(
            model=llm_model_name,
            api_key=openai_api_key,
            request_timeout=300,
        )
    else:
        raise ValueError(f"Unsupported llm provider: {llm_provider}")

    # 7. Response synthesizer with custom prompt
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=prompt_template,
    )

    # 8. Query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker] if reranker else None,
        response_synthesizer=response_synthesizer,
    )

    return query_engine, client