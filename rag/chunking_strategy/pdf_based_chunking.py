from .base import ChunkingStrategy


class PdfBasedChunking(ChunkingStrategy):
    strategy_name = "pdf_based_chunking"

    def chunk(self, text: str) -> list[str]:
        return [text]
