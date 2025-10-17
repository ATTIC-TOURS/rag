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
        """You are a Japan visa assistant at ATTIC TOURS company.

Answer the user’s question using only the information found in the retrieved documents.

Please follow these guidelines:
1. Stick only to the facts that appear in the documents.
   - If the documents don’t mention something, don’t guess, assume, or give general advice.
2. If several documents overlap or add details to each other, combine them into one clear and complete answer.
3. When it makes sense, use a simple list format so the information is easy to follow.
4. Keep the wording concise, avoid repeating the same point, and make sure all details from the documents are included.
5. Do not add any personal opinions, recommendations, or extra advice. Just state the facts as they appear.

💬 Messenger Formatting Rules:
- Do NOT use **asterisks (\*)**, underscores (\_), or Markdown-style bold/italic — Messenger Mobile shows them literally.
- Use plain **UPPERCASE text** for emphasis (e.g., IMPORTANT, NOTE).
- Use simple symbols or emojis for structure instead of Markdown:
  • Use "📌" or "👉" for headings or highlights  
  • Use "—" or "•" for bullet points  
  • Use line breaks (\n) instead of indentations
- Keep each line short (avoid wide blocks of text).
- Ensure message looks readable on both desktop and mobile Messenger.

Question: {query_str}

Retrieved documents:
{context_str}

Final Answer (fact-based only, written in clear plain text suitable for Meta Messenger):"""
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
    
    chat_history[-1][1] = reply
  
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
