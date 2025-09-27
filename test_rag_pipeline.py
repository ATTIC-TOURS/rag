from rag.rag_pipeline import build_two_stage_rag_pipeline
from llama_index.core.prompts import PromptTemplate
import time
from dotenv import load_dotenv
from rag.indexing.modules import set_global_embeddings
import asyncio

load_dotenv()
hf_embeddings_name = "intfloat/multilingual-e5-base"
set_global_embeddings(model_name=hf_embeddings_name, provider="hf")

params = {
        "index_name": "Custom_splitter_w_context_hf",
        "alpha": 0.8,
        "similarity_top_k": 10,
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        "rerank_top_n": 3,
        "fact_prompt": PromptTemplate(
            "Extract only factual statements from the documents below.\n"
            "Write each fact as a short, atomic bullet point.\n"
            "Do not add or guess anything.\n\n"
            "Question: {query_str}\n"
            "Documents:\n"
            "{context_str}\n\n"
            "Answer (facts only, bullet points):"
        ),
        "answer_prompt": PromptTemplate(
            "You are a helpful assistant for Japan visa questions.\n"
            "Use ONLY the provided facts below to answer clearly and concisely.\n"
            "If multiple conditions exist, explain them separately.\n"
            "Do not invent or add information.\n\n"
            "Facts:\n"
            "{context_str}\n\n"
        ),
    }

async def main():
        
    query_engine, client = await build_two_stage_rag_pipeline(**params)

    start = time.time()  # record start time

    response = await query_engine.query("Ano po ang requirements para po sa transit??")
    print(response)

    end = time.time()  # record end time

    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")

asyncio.run(main())