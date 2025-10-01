from contextlib import asynccontextmanager
from fastapi import APIRouter, Request, Response, BackgroundTasks, Query
from dotenv import load_dotenv
import os
import httpx
from weaviate import WeaviateAsyncClient
from llama_index.core.prompts import PromptTemplate
from ..rag.rag_pipeline import build_rag_pipeline
from ..rag.indexing.modules import set_global_embeddings

# --- Load env ---
load_dotenv()
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFICATION_TOKEN")
FB_SEND_API_URL = "https://graph.facebook.com/v23.0/me/messages"

# --- Globals ---
query_engine: any
client: WeaviateAsyncClient

# --- Params ---
params = {
    "index_name": "Custom_splitter_w_context_hf",
    "alpha": 0.8,
    "base_k": 5,
    "expansion_k": 5,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rerank_top_n": 5,
    "fact_prompt": PromptTemplate(
        """You are a Japan visa assistant.

Your task is to answer the user's question using ONLY the retrieved documents. 
Follow these strict rules:
1. Use ONLY facts explicitly stated in the retrieved documents. 
   - If something is not mentioned, DO NOT invent, assume, or guess.
2. When multiple documents provide overlapping or complementary information, merge them into a single clear answer without losing details.
3. Present the answer in a structured, list-like format when appropriate (to maximize coverage of factual details).
4. Be concise and avoid repetition, but ensure completeness of the retrieved facts.
5. Do not add any recommendations.

Question: {query_str}

Retrieved documents:
{context_str}

Final Answer:"""
    ),
    "two_stage": False,
    "use_query_expansion": True,
    "query_expansion_num": 2,
}

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: APIRouter):
    set_global_embeddings(model_name="intfloat/multilingual-e5-base", provider="hf")
    global query_engine, client
    print("🚀 Initializing RAG pipeline...")
    query_engine, client = await build_rag_pipeline(**params)
    yield
    print("🛑 Shutting down RAG pipeline...")
    await client.close()

router = APIRouter(lifespan=lifespan, prefix="/chat", tags=["messenger"])

# --- Messenger API helpers ---
async def send_typing_on(recipient_id: str):
    payload = {"recipient": {"id": recipient_id}, "sender_action": "typing_on"}
    async with httpx.AsyncClient() as client:
        r = await client.post(FB_SEND_API_URL, params={"access_token": PAGE_ACCESS_TOKEN}, json=payload)
        if r.status_code != 200:
            print("❌ Typing On Error:", r.status_code, r.text)

async def send_typing_off(recipient_id: str):
    payload = {"recipient": {"id": recipient_id}, "sender_action": "typing_off"}
    async with httpx.AsyncClient() as client:
        r = await client.post(FB_SEND_API_URL, params={"access_token": PAGE_ACCESS_TOKEN}, json=payload)
        if r.status_code != 200:
            print("❌ Typing Off Error:", r.status_code, r.text)

MAX_LEN = 1800

def split_text(text: str, max_len: int = MAX_LEN):
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

async def send_message(recipient_id: str, text: str):
    if not text.strip():
        return
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    async with httpx.AsyncClient() as client:
        r = await client.post(FB_SEND_API_URL, params={"access_token": PAGE_ACCESS_TOKEN}, json=payload)
        if r.status_code != 200:
            print("❌ FB Send Error:", r.status_code, r.text)

async def send_long_message(recipient_id: str, text: str):
    for part in split_text(text):
        await send_message(recipient_id, part)

# --- Webhook verify ---
@router.get("/messenger-webhook")
async def verify_meta(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return Response(content=hub_challenge, media_type="text/plain")
    return Response(content="Invalid verification token", status_code=403)

# --- Core processing ---
async def process_user_message(recipient_id: str, text: str):
    print(f"[REAL] User {recipient_id} said: {text}")
    try:
        # Show typing
        await send_typing_on(recipient_id)

        # Ask RAG
        reply_text = await inquire(text)

        # Send reply in chunks if needed
        await send_long_message(recipient_id, reply_text)
    finally:
        # Turn typing bubble off
        await send_typing_off(recipient_id)

@router.post("/messenger-webhook")
async def messenger_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    print("Webhook received:", data)

    if "entry" in data:
        for entry in data["entry"]:
            for messaging in entry.get("messaging", []):
                sender_id = messaging["sender"]["id"]

                if "message" in messaging:
                    # Skip bot echo messages to avoid loops
                    if messaging["message"].get("is_echo"):
                        continue

                    text = messaging["message"].get("text", "")
                    if text:  # only process real text
                        background_tasks.add_task(process_user_message, sender_id, text)

    return {"status": "ok"}

# --- RAG Inquire ---
@router.post("/japan-visa-inquire")
async def inquire(message: str) -> str:
    response = await query_engine.query(message)
    return response["final_answer"]
