from llama_index.core.embeddings import BaseEmbedding
# from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings

# <-- Add this import
from typing import Any

class MyEmbeddings(BaseEmbedding):
    _model: Any  # Accepts SentenceTransformer, OpenAIEmbeddings, or HuggingFaceAPIEmbeddings

    def __init__(self, model):
        super().__init__()
        self._model = model

    def _get_text_embedding(self, text: str):
        # if isinstance(self._model, SentenceTransformer):
        #     return self._model.encode(f"passage: {text}").tolist()

        if isinstance(self._model, OpenAIEmbeddings):
            return self._model.embed_documents([text])[0]

        elif hasattr(self._model, "embed"):  # ✅ HuggingFace API client style
            return self._model.embed(f"passage: {text}")

        else:
            raise ValueError("Unsupported embedding model")

    def _get_query_embedding(self, query: str):
        # if isinstance(self._model, SentenceTransformer):
        #     return self._model.encode(f"query: {query}").tolist()

        if isinstance(self._model, OpenAIEmbeddings):
            return self._model.embed_query(query)

        elif hasattr(self._model, "embed"):  # ✅ HuggingFace API client style
            return self._model.embed(f"query: {query}")

        else:
            raise ValueError("Unsupported embedding model")

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)
