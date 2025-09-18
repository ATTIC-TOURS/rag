from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class MyEmbeddings(BaseEmbedding):
    _model: any = PrivateAttr()

    def __init__(self, model):
        super().__init__()
        self._model = model  # HuggingFace model

    def _get_text_embedding(self, text: str):
        return self._model.encode(text).tolist()

    def _get_query_embedding(self, query: str):
        return self._model.encode(query).tolist()

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)