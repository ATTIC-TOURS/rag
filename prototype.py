import gradio as gr
import asyncio
import nest_asyncio
from llama_index.core.prompts import PromptTemplate
from rag.indexing.modules import set_global_embeddings
from rag.rag_pipeline import build_rag_pipeline
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# --- RAG Params ---
params = {
    "index_name": "Custom_splitter_w_context_hf",
    "alpha": 0.8,
    "base_k": 5,
    "expansion_k": 5,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "rerank_top_n": 5,
    "fact_prompt": PromptTemplate(
        """You are a Japan visa assistant at ATTIC TOURS.

Your task is to answer the user's question using ONLY the retrieved documents.

Rules:
1. Use ONLY facts explicitly stated in the retrieved documents.
2. Merge overlapping info into one answer.
3. Use structured format when possible (bullet points, numbered lists).
4. Be concise but complete.
5. Do NOT add recommendations.

Question: {query_str}

Retrieved documents:
{context_str}

Final Answer (Markdown format):"""
    ),
    "two_stage": False,
    "use_query_expansion": True,
    "query_expansion_num": 2,
}

set_global_embeddings(model_name="intfloat/multilingual-e5-base", provider="hf")

# --- Lazy RAG initialization ---
query_engine = None

async def get_query_engine():
    global query_engine
    if query_engine is None:
        query_engine, _ = await build_rag_pipeline(**params)
    return query_engine

# --- Async chatbot logic with typing bubble ---
async def respond(message, chat_history):
    qe = await get_query_engine()
    chat_history = chat_history or []

    # 1. Add user message
    chat_history.append([message, ""])
    yield "", chat_history

    # 2. Add "typing..." bubble for assistant
    chat_history.append(["", "🤖 typing..."])
    yield "", chat_history

    # 3. Query RAG
    result = await qe.aquery(message)
    reply = result["final_answer"]

    # 4. Remove typing bubble and add empty assistant message for streaming
    chat_history.pop()  # remove "typing..." bubble
    chat_history.append(["", ""])
    
    # 5. Stream bot reply word by word
    bot_reply = ""
    for word in reply.split():
        bot_reply += word + " "
        chat_history[-1][1] = bot_reply.strip()
        yield "", chat_history  # matches [Textbox, Chatbot]

# --- Launch UI ---
def run_prototype():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="Japan Visa Attic Tours Chatbot")
        msg = gr.Textbox(placeholder="Ask about Japan visa...", show_label=False)
        clear = gr.ClearButton([msg, chatbot])

        # Bind submit to async streaming function
        msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)

    demo.launch(share=True)

if __name__ == "__main__":
    run_prototype()
