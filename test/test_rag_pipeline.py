from rag.rag_pipeline import build_rag_pipeline
from llama_index.core.prompts import PromptTemplate
from rag.my_decorators.time import time_performance
from rag.indexing.modules import set_global_embeddings
import asyncio
from colorama import init, Fore
init(autoreset=True)


params = {
    "index_name": "JapanVisa",
    "alpha": 0.8,
    "base_k": 5,
    "expansion_k": 5,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_top_n": 5,
    "fact_prompt": PromptTemplate(
        """You are a Japan visa assistant at ATTIC TOURS company.

You must answer the user’s question using ONLY the information found in the retrieved documents.

Please follow these guidelines:

1. Stick strictly to the facts that appear in the retrieved documents.  
   - If the documents don’t mention something, do NOT guess, assume, or give general advice.

2. If several documents overlap or add details to each other, combine them into one clear and complete answer.

3. When it makes sense, use a simple list format so the information is easy to follow.

4. Keep wording concise. Avoid repeating points and ensure all relevant details are included.

5. Do NOT add any personal opinions, recommendations, or extra advice. Only state the facts as they appear.

6. Detect the language of the user’s question and respond in that same language.  
   - If the user writes in Japanese, reply in Japanese.  
   - If the user writes in English, reply in English.  
   - If the user mixes languages (e.g., Taglish or Japanglish), choose the dominant language.

💬 Messenger Formatting Rules:
- Do NOT use asterisks (*), underscores (_), or Markdown bold/italic — Messenger Mobile shows them literally.
- Use plain UPPERCASE text for emphasis (e.g., IMPORTANT, NOTE).
- Use simple symbols or emojis for structure instead of Markdown:
  • Use "📌" or "👉" for headings or highlights  
  • Use "—" or "•" for bullet points  
  • Use line breaks (\n) instead of indentations
- Keep each line short (avoid wide blocks of text).
- Ensure the message looks readable on both desktop and mobile Messenger.

Question: {query_str}

Retrieved documents:
{context_str}

Final Answer (fact-based only, written clearly in the same language as the user query, formatted for Meta Messenger):



"""
    ),
    "two_stage": False,
    "use_query_expansion": True,
    "query_expansion_num": 1,
}


@time_performance("main")
async def main():
    hf_embeddings_name = "intfloat/multilingual-e5-base"
    provider = "hf"
    set_global_embeddings(model_name=hf_embeddings_name, provider=provider)
    query_engine, client = await build_rag_pipeline(**params, is_cloud_storage=True)

    while True:
        print("type \"quit\" if you wish to quit")
        query = input("query: ")
        
        if query.strip().lower() == "quit":
            break
        
        response = await query_engine.query(query)
        print(Fore.GREEN + response["final_answer"])
    
    await client.close()


asyncio.run(main())
