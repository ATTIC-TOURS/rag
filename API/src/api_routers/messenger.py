# ------------------------- DEPENDENCIES ------------------------- #
import os
from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

from fastapi import APIRouter, Request, Response, BackgroundTasks, Query

import httpx

from weaviate import WeaviateAsyncClient

from llama_index.core.prompts import PromptTemplate

import sqlite3

import asyncio

from colorama import init, Fore

init(autoreset=True)

from ..rag.rag_pipeline import build_rag_pipeline

# ------------------------- DEPENDENCIES (END) ------------------------- #

# ------------------------- GLOBAL VARIABLES ------------------------- #
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFICATION_TOKEN")
FB_SEND_API_URL = "https://graph.facebook.com/v23.0/me/messages"

# --- Globals ---
query_engine: any
client: WeaviateAsyncClient

params = {
    "index_name": "JapanVisaDemo",
    "alpha": 0.8,
    "base_k": 5,
    "use_query_expansion": True,
    "query_expansion_num": 1,
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
}
# ------------------------- GLOBAL VARIABLES (END) ------------------------- #


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: APIRouter):
    global query_engine, client
    print("🚀 Initializing RAG pipeline...")
    query_engine, client = await build_rag_pipeline(**params, is_cloud_storage=True)
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
        try:
            message_id = r.json().get("message_id")
        except Exception:
            message_id = None

        return message_id


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
        # await send_mark_seen_indicator(recipient_id)

        # await send_message(recipient_id, "💬")

        # await send_typing_indicator(recipient_id)

        await asyncio.gather(
            send_mark_seen_indicator(recipient_id),
            send_typing_indicator(recipient_id),
        )

        # Ask RAG
        reply_text = await inquire(text)

        await send_typing_off(recipient_id)

        # Send reply in chunks if needed
        await send_long_message(recipient_id, reply_text)

    except Exception as e:
        print(f"❌ Error in process_user_message: {e}")
        await send_typing_off(recipient_id)


def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY
        )
    """
    )
    conn.commit()
    conn.close()


def user_exists(user_id: str) -> bool:
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def save_user(user_id: str):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()


greeting_text = (
    "👋 Hi there! Welcome to Attic Tours — your trusted assistant for Japan Visa inquiries.\n\n"
    "I’m an automated chatbot ready to help you with visa requirements, processing fees, application locations, contact numbers, and more. 🇯🇵\n\n"
    "💬 You can ask me anything — no need to follow strict sentences!\n"
    "Feel free to message in English, Filipino, or even Taglish.\n\n"
    "How can I assist you today?"
)


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
                        if not user_exists(sender_id):
                            # Save and greet the new user
                            save_user(sender_id)
                            await send_message(sender_id, greeting_text)
                        else:
                            background_tasks.add_task(
                                process_user_message, sender_id, text
                            )

    return {"status": "ok"}


# --- RAG Inquire ---
@router.post("/japan-visa-inquire")
async def inquire(message: str) -> str:
    response = await query_engine.query(message)
    return response["final_answer"]


init_db()
