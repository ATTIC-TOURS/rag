from fastapi import APIRouter, Path, Body
from typing import Annotated

router = APIRouter(prefix="/knowledge", tags=["knowledge base"])


@router.get("/japan-visa-documents")
async def read_japan_visa_documents():
    pass


@router.get("/japan-visa-documents/{doc_id}")
async def read_japan_visa_document(doc_id: Annotated[int, Path()]):
    pass


@router.post("/japan-visa-documents")
async def add_japan_visa_document(document: Annotated[str, Body()]):
    pass


@router.delete("/japan-visa-documents/{doc_id}")
async def remove_japan_visa_document(doc_id: Annotated[int, Path()]):
    pass
