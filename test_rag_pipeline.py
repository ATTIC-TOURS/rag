from rag.rag_pipeline import build_rag_pipeline
from llama_index.core.prompts import PromptTemplate
import time
from dotenv import load_dotenv
from rag.indexing.modules import set_global_embeddings


load_dotenv()

index_names = [
    "Custom_splitter_hf",
    "Custom_splitter_openai",
    "Custom_splitter_w_context_hf",
    "Custom_splitter_w_context_openai",
    "Normal_splitter_hf",
    "Normal_splitter_openai",
    "Normal_splitter_w_context_hf",
    "Normal_splitter_w_context_openai",
]

hf_embeddings_name = "intfloat/multilingual-e5-base"
openai_embeddings_name = "text-embedding-3-small"
set_global_embeddings(model_name=openai_embeddings_name, provider="openai")

query_engine, client = build_rag_pipeline(
    index_name=index_names[1], #ok
    similarity_top_k=1, #ok
    alpha=0.8, #ok
    cross_encoder_model=None, #ok
    rerank_top_n=1,  # ok
    llm_provider="openai",  # "ollama" or "openai" # ok
    llm_model_name="gpt-4o-mini",  # ok
    prompt_template=PromptTemplate(  # ok
        "You are a helpful assistant that answers Japan visa questions.\n\n"
        "Question: {query_str}\n\n"
        "Here are the retrieved documents:\n{context_str}\n\n"
        "Answer clearly and concisely."
    ),
)

start = time.time()  # record start time

response = query_engine.query("Ano po ang requirements para po sa transit??")
print(response)

end = time.time()  # record end time

elapsed = end - start
print(f"Elapsed time: {elapsed:.2f} seconds")

client.close()
