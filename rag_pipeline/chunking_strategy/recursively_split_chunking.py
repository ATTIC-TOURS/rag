from .base import ChunkingStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursivelySplitChunking(ChunkingStrategy):
    strategy_name = "recursively_split_chunking"

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # max characters per chunk
            chunk_overlap=self.chunk_overlap,  # overlap to preserve context
            separators=["\n\n", "\n", " ", ""],  # order of splitting paragraphs->sentences->spaces->characters
        )
        
        return text_splitter.split_text(text)
