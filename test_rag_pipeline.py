from rag.rag_pipeline import build_rag_pipeline
from llama_index.core.prompts import PromptTemplate
import time
from dotenv import load_dotenv
from rag.indexing.modules import set_global_embeddings
import asyncio

load_dotenv()
hf_embeddings_name = "intfloat/multilingual-e5-base"
set_global_embeddings(model_name=hf_embeddings_name, provider="hf")

# PromptTemplate(
#             "Extract only factual statements from the documents below.\n"
#             "Write each fact as a short, atomic bullet point.\n"
#             "Do not add or guess anything.\n\n"
#             "Question: {query_str}\n"
#             "Documents:\n"
#             "{context_str}\n\n"
#             "Answer (facts only, bullet points):"
#         )

params = {
    "index_name": "Custom_splitter_w_context_hf",
    "alpha": 0.8,
    "similarity_top_k": 10,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "rerank_top_n": 5,
    "fact_prompt": PromptTemplate(
        "You are a helpful assistant that answers Japan visa questions.\n\n"
        "You must follow these rules:\n"
        "1. Use ONLY the information in the retrieved documents below.\n"
        "2. Do NOT add any information that is not explicitly supported by the documents.\n"
        "3. Provide a complete answer, but concise. Cover all relevant details from the documents without adding extra assumptions.\n\n"
        "Question: {query_str}\n\n"
        "Retrieved documents:\n{context_str}\n\n"
        "Answer:"
    ),
    # "answer_prompt": PromptTemplate(
    #     "You are a helpful assistant for Japan visa questions.\n"
    #     "Use ONLY the provided facts below to answer clearly and concisely.\n"
    #     "If multiple conditions exist, explain them separately.\n"
    #     "Do not invent or add information.\n\n"
    #     "Facts:\n"
    #     "{context_str}\n\n"
    # ),
    "two_stage": False,
}


async def main():

    query_engine, client = await build_rag_pipeline(**params)

    start = time.time()  # record start time

    response = await query_engine.query("Ano po ang requirements para po sa transit??")
    print(response)

    end = time.time()  # record end time

    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")
    
    await client.close()


asyncio.run(main())
