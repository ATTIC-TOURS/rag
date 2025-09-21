from rag_pipeline.rag_pipeline import build_rag_pipeline
from llama_index.core.prompts import PromptTemplate
import time
from dotenv import load_dotenv
import os


load_dotenv()
OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")

query_engine, client = build_rag_pipeline(
    index_name="Requirements",
    embeddings_model_name="intfloat/multilingual-e5-base",
    similarity_top_k=20,
    alpha=0.8,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n=3, 
    llm_provider='ollama', # "ollama" or "openai"
    llm_model_name="gemma3:1b",
    openai_api_key=OPEN_API_KEY,
    prompt_template=PromptTemplate(
        "You are a helpful assistant that answers Japan visa questions.\n\n"
        "Question: {query_str}\n\n"
        "Here are the retrieved documents:\n{context_str}\n\n"
        "Answer clearly and concisely."
    )
)

start = time.time()  # record start time

response = query_engine.query("What are the requirements for a Japan tourist visa?")
print(response)

end = time.time()  # record end time
    
elapsed = end - start
print(f"Elapsed time: {elapsed:.2f} seconds")

client.close()
