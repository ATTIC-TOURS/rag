from chunking_strategy.base import ChunkingStrategy


class PdfBasedChunking(ChunkingStrategy):
    strategy_name = "pdf_based_chunking"

    def __init__(self):
        super().__init__()
        
    def chunk(self, text: str) -> list[str]:
        return [text]
