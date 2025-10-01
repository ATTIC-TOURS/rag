from contextlib import asynccontextmanager
from fastapi import APIRouter, Request, Response, Body
from typing import Annotated
from ..models.messenger import Message
from ..rag.rag_pipeline import build_rag_pipeline
from dotenv import load_dotenv
import os
import asyncio

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import requests
from weaviate import WeaviateClient
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

load_dotenv()
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")  # Or set directly
VERIFY_TOKEN = os.getenv("VERIFICATION_TOKEN")
FB_SEND_API_URL = "https://graph.facebook.com/v17.0/me/messages"
query_engine: RetrieverQueryEngine
client: WeaviateClient
from ..rag.indexing.modules import set_global_embeddings

selected_params = {
    "index_name": "Custom_splitter_w_context_hf",
    "alpha": 1.0,
    "similarity_top_k": 3,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "prompt_template": PromptTemplate(
        "You are a helpful assistant that answers Japan visa questions.\n\n"
        "You must follow these rules:\n"
        "1. Use ONLY the information in the retrieved documents below.\n"
        "2. Do NOT add any information that is not explicitly supported by the documents.\n"
        "3. Provide a complete answer, but concise. Cover all relevant details from the documents without adding extra assumptions.\n\n"
        "Question: {query_str}\n\n"
        "Retrieved documents:\n{context_str}\n\n"
        "Answer:"
    ),
}
fixed_params = {
    "llm_model_name": "gpt-5",
    "rerank_top_n": 3,
    "llm_provider": "openai",
}


@asynccontextmanager
async def lifespan(app: APIRouter):
    set_global_embeddings(
        **{
            "model_name": "intfloat/multilingual-e5-base",
            "provider": "hf",
        }  # type: ignore
    )
    global query_engine, client
    print("🚀 Initializing RAG pipeline...")
    query_engine, client = build_rag_pipeline(**selected_params, **fixed_params)
    yield
    print("🛑 Shutting down RAG pipeline...")
    # You can clean up client connections here if needed
    client.close()


router = APIRouter(lifespan=lifespan, prefix="/chat", tags=["messenger"])


@router.get("/messenger-webhook")
async def verify_meta(request: Request):
    params = dict(request.query_params)
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        # Return the challenge as plain text, not int or JSON
        return Response(content=params["hub.challenge"], media_type="text/plain")
    return Response(content="Invalid verification token", status_code=403)


async def send_message(recipient_id: str, text: str):

    text = await inquire(Message(**{"text": text}))  # type: ignore
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    params = {"access_token": PAGE_ACCESS_TOKEN}
    response = requests.post(FB_SEND_API_URL, params=params, json=payload)
    print("Sent message response:", response.json())


@router.post("/messenger-webhook")
async def messenger_webhook(request: Request):
    data = await request.json()
    print("Webhook received:", data)

    if "entry" in data:
        for entry in data["entry"]:
            for messaging in entry.get("messaging", []):
                sender_id = messaging["sender"]["id"]
                payload = {
                    "recipient": {"id": sender_id},
                    "sender_action": "typing_on",
                    "messaging_type": "RESPONSE",
                }
                requests.post(
                    FB_SEND_API_URL,
                    params={"access_token": PAGE_ACCESS_TOKEN},
                    json=payload,
                )

                if "message" in messaging:
                    text = messaging["message"].get("text", "")
                    print(f"[REAL] User {sender_id} said: {text}")
                    await asyncio.sleep(1.5)
                    # Automated reply
                    reply_text = f"You said: {text}"
                    # await send_message(sender_id, reply_text)

    return {"status": "ok"}


@router.post("/japan-visa-inquire")
async def inquire(message: Annotated[Message, Body()]):

    return str(query_engine.query(message.text))
