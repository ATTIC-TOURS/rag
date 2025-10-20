# ------------------------- DEPENDENCIES ------------------------- #
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
# from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

# from huggingface_hub import InferenceClient

import weaviate, hashlib
from weaviate.classes.init import Auth

# from sentence_transformers import SentenceTransformer

from langchain_openai import OpenAIEmbeddings

from colorama import Fore

from .my_decorators.time import time_performance
from .embeddings.embeddings import MyEmbeddings

# ------------------------- DEPENDENCIES (END) ------------------------- #

connection_config = {
    "local": {"port": 8080, "grpc_port": 50051, "skip_init_checks": True},
    "cloud": {
        "cluster_url": os.environ["WEAVIATE_URL"],
        "auth_credentials": Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
    },
}


class HuggingFaceAPIEmbeddings:
    def __init__(self, model_name: str):
        # self.model = model_name
        # self.client = InferenceClient(
        #     provider="hf-inference",
        #     api_key=os.environ["HF_API_KEY"],
        # )
        pass

    def embed(self, text: str):
        # result = self.client.feature_extraction(
        #     text=text,
        #     model="intfloat/multilingual-e5-base",
        # )
        # print(Fore.GREEN + f"{result}")
        # return result
        pass


import cohere
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import List, Optional
from pydantic import Field, PrivateAttr 
from llama_index.core.schema import NodeWithScore, QueryBundle 

# Note: BaseNodePostprocessor might already inherit from BaseModel
class CohereReranker(BaseNodePostprocessor):
    # ... (Pydantic Fields and __init__ are correct from the previous step)
    model: str = Field(default="rerank-english-v3.0", description="The Cohere rerank model to use.")
    top_n: int = Field(default=5, description="The number of top documents to return.")
    api_key: Optional[str] = Field(default=None, description="Your Cohere API key.")

    _client: cohere.Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = cohere.Client(api_key=self.api_key)


    def _postprocess_nodes(
        self, 
        nodes: List[NodeWithScore], 
        query_bundle: Optional[QueryBundle] = None  # <-- Changed argument name and added type hint
    ) -> List[NodeWithScore]:
        """
        Reranks the nodes using the Cohere API.
        
        The LlamaIndex framework calls this method with the query as a QueryBundle.
        """
        # The user's query string is now guaranteed to be in the query_bundle object
        if query_bundle is None:
            raise ValueError("QueryBundle must be provided for reranking.")

        # 🛑 CRITICAL FIX: Extract the actual query string from the QueryBundle
        query_str = query_bundle.query_str
        
        if not query_str:
            return nodes # Return original nodes if query is empty

        # 1. Extract document texts
        docs: List[str] = [node.text for node in nodes]
        
        # 2. Call the Cohere Rerank API
        response = self._client.rerank(
            model=self.model,
            query=query_str, # <-- Now passing a guaranteed string
            documents=docs,
            top_n=self.top_n,
        )

        # 3. Create a list of the top N reranked nodes with their new scores
        reranked_nodes: List[NodeWithScore] = []
        for result in response.results:
            original_node = nodes[result.index]
            # Use the score from Cohere's relevance score
            reranked_node = NodeWithScore(
                node=original_node.node, 
                score=result.relevance_score
            )
            reranked_nodes.append(reranked_node)
            
        return reranked_nodes


@time_performance("set_global_embeddings")
def set_global_embeddings(model_name: str, provider: str):
    if provider == "hf":
        model = HuggingFaceAPIEmbeddings(model_name=model_name)
        # model = SentenceTransformer(model_name)
    elif provider == "openai":
        model = OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError("Provider must be 'hf' or 'openai'")

    Settings.embed_model = MyEmbeddings(model)

    print("Global embeddings set to model: ", end="")
    print(Fore.BLUE + f"{provider}:{model_name}")


# --- Query expansion helper ---
@time_performance(name="expand_query")
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
@time_performance(name="deduplicate_nodes")
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
@time_performance("build_rag_pipeline")
async def build_rag_pipeline(
    index_name: str,
    alpha: float,
    cross_encoder_model: str | None = None,
    rerank_top_n: int = 5,  # final cut
    fact_prompt: PromptTemplate | None = None,
    use_query_expansion: bool = False,
    query_expansion_num: int = 3,
    base_k: int = 10,  # docs for original query
    expansion_k: int = 5,  # docs for each expansion
    is_cloud_storage: bool = False,
):

    set_global_embeddings(model_name="text-embedding-3-small", provider="openai")

    # 1. Connect to Weaviate
    client = None
    if is_cloud_storage:
        client = weaviate.use_async_with_weaviate_cloud(**connection_config["cloud"])
    else:
        client = weaviate.use_async_with_local(**connection_config["local"])
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
        reranker = CohereReranker(
            api_key=os.environ["COHERE_TRIAL_KEY"],
            model="rerank-english-v3.0",
            top_n=rerank_top_n,
        )

    # 4. Synthesizers
    llm = OpenAI(model="gpt-5-chat-latest", request_timeout=300)
    fact_synthesizer = get_response_synthesizer(llm=llm, text_qa_template=fact_prompt)

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
                query_bundle = QueryBundle(query_str=query)
                unique_nodes = reranker.postprocess_nodes(
                    query_bundle=query_bundle, nodes=unique_nodes
                )

            # Keep only top N
            final_nodes = unique_nodes[:rerank_top_n]

            # Step 4: fact synthesis
            fact_response = await fact_synthesizer.asynthesize(query, final_nodes)

            # Step 5: final synthesis (optional)

            return {
                "expanded_queries": expanded_queries,
                "retrieved_docs": len(all_nodes),
                "unique_docs": len(unique_nodes),
                "final_answer": str(fact_response),
                "facts": fact_response,
            }

        @time_performance("query")
        async def query(self, query: str):
            return await self.aquery(query)

    return FlexibleQueryEngine(), client
