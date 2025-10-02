from contextlib import asynccontextmanager
from fastapi import APIRouter, Request, Response, BackgroundTasks, Query
from dotenv import load_dotenv
import os
import httpx
from weaviate import WeaviateAsyncClient
from llama_index.core.prompts import PromptTemplate
from ..rag.rag_pipeline import build_rag_pipeline
from ..rag.indexing.modules import set_global_embeddings

from colorama import init, Fore

init(autoreset=True)

# --- Load env ---
load_dotenv()
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFICATION_TOKEN")
FB_SEND_API_URL = "https://graph.facebook.com/v21.0/me/messages"

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
        """You are a Japan visa assistant at ATTIC TOURS company.

Answer the user’s question using only the information found in the retrieved documents. 
Please follow these guidelines:
1. Stick only to the facts that appear in the documents.  
   - If the documents don’t mention something, don’t guess, assume, or give general advice.  
2. If several documents overlap or add details to each other, combine them into one clear and complete answer.  
3. When it makes sense, use a simple list format so the information is easy to follow.  
4. Keep the wording concise, avoid repeating the same point, and make sure all details from the documents are included.  
5. Do not add any personal opinions, recommendations, or extra advice. Just state the facts as they appear.  

Question: {query_str}

Retrieved documents:  
{context_str}

Final Answer (fact-based only, written in a clear style suitable for Meta Messenger):
"""
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

async def send_mark_seen_indicator(sender_id: str):

    payload = {"recipient": {"id": sender_id}, "sender_action": "mark_seen"}

    # CRITICAL: Added detailed logging for Facebook API errors
    print(f"Attempting to send mark_seen to {sender_id} via {FB_SEND_API_URL}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                FB_SEND_API_URL,
                params={"access_token": PAGE_ACCESS_TOKEN},
                json=payload,
            )
            # This will raise an exception for 4xx or 5xx status codes
            response.raise_for_status()
            print(f"{Fore.GREEN}SUCCESS: Mark seen indicator sent.")

        except httpx.HTTPStatusError as e:
            # Handle specific HTTP errors returned by the API
            print(f"{Fore.RED}CRITICAL FB API ERROR during typiMark seenng_on:")
            print(f"{Fore.RED}  Status: {e.response.status_code}")
            # The body contains the detailed error JSON from Facebook (e.g., auth failure)
            print(f"{Fore.RED}  Body: {e.response.text}")
        except httpx.RequestError as e:
            # Handle network errors (DNS, connection issues)
            print(f"{Fore.RED}NETWORK ERROR during Mark seen: {e}")
        except Exception as e:
            # Catch all other unexpected errors
            print(f"{Fore.RED}UNEXPECTED ERROR during Mark seen: {e}")

# --- Messenger API helpers ---
async def send_typing_indicator(sender_id: str):

    payload = {"recipient": {"id": sender_id}, "sender_action": "typing_on"}

    # CRITICAL: Added detailed logging for Facebook API errors
    print(f"Attempting to send typing_on to {sender_id} via {FB_SEND_API_URL}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                FB_SEND_API_URL,
                params={"access_token": PAGE_ACCESS_TOKEN},
                json=payload,
            )
            # This will raise an exception for 4xx or 5xx status codes
            response.raise_for_status()
            print(f"{Fore.GREEN}SUCCESS: Typing indicator sent.")

        except httpx.HTTPStatusError as e:
            # Handle specific HTTP errors returned by the API
            print(f"{Fore.RED}CRITICAL FB API ERROR during typing_on:")
            print(f"{Fore.RED}  Status: {e.response.status_code}")
            # The body contains the detailed error JSON from Facebook (e.g., auth failure)
            print(f"{Fore.RED}  Body: {e.response.text}")
        except httpx.RequestError as e:
            # Handle network errors (DNS, connection issues)
            print(f"{Fore.RED}NETWORK ERROR during typing_on: {e}")
        except Exception as e:
            # Catch all other unexpected errors
            print(f"{Fore.RED}UNEXPECTED ERROR during typing_on: {e}")


async def send_typing_off(recipient_id: str):
    payload = {"recipient": {"id": recipient_id}, "sender_action": "typing_off"}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            FB_SEND_API_URL, params={"access_token": PAGE_ACCESS_TOKEN}, json=payload
        )
        if r.status_code != 200:
            print("❌ Typing Off Error:", r.status_code, r.text)


MAX_LEN = 1800


def split_text(text: str, max_len: int = MAX_LEN):
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


async def send_message(recipient_id: str, text: str):
    if not text.strip():
        return
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            FB_SEND_API_URL, params={"access_token": PAGE_ACCESS_TOKEN}, json=payload
        )
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
        # Show mark seen
        await send_mark_seen_indicator(recipient_id)
        
        await send_message(recipient_id, "🤖 typing...")
        
        await send_typing_indicator(recipient_id)

        # Ask RAG
        reply_text = await inquire(text)

        # Send reply in chunks if needed
        await send_long_message(recipient_id, reply_text)
    finally:
        # Turn typing bubble off
        await send_typing_off(recipient_id)


@router.post("/messenger-webhook")
async def handle_messenger_webhook(request: Request, background_tasks: BackgroundTasks):
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
