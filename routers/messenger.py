from contextlib import asynccontextmanager
from fastapi import APIRouter, Request, Response, Body
from typing import Annotated
from ..models.messenger import Message
from ..rag_pipeline.rag_pipeline import RagPipeline
from dotenv import load_dotenv
import os
import requests

load_dotenv()
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")  # Or set directly
VERIFY_TOKEN = os.getenv("VERIFICATION_TOKEN")
FB_SEND_API_URL = "https://graph.facebook.com/v17.0/me/messages"
rag_pipeline: RagPipeline


@asynccontextmanager
async def lifespan(app: APIRouter):
    # Load the pipeline
    global rag_pipeline
    rag_pipeline = RagPipeline()
    yield


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

    text = await inquire(Message(**{"text": text}))
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
                payload = {"recipient": {"id": sender_id}, "sender_action": "typing_on"}
                requests.post(
                    FB_SEND_API_URL,
                    params={"access_token": PAGE_ACCESS_TOKEN},
                    json=payload,
                )
                if "message" in messaging:
                    text = messaging["message"].get("text", "")
                    print(f"[REAL] User {sender_id} said: {text}")
                    # Automated reply
                    reply_text = f"You said: {text}"
                    await send_message(sender_id, reply_text)

    return {"status": "ok"}


@router.post("/japan-visa-inquire")
async def inquire(message: Annotated[Message, Body()]):
    response = ""
    for text in rag_pipeline.answer(message.text):
        response += text

    return response
