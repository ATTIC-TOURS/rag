from chunking_strategy.base import ChunkingStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PdfBasedRecursivelySplitChunking(ChunkingStrategy):
    strategy_name = "pdf_based_recursively_split_chunking"

    def __init__(self, pdf_num_split: int, chunk_overlap_rate: float):
        super().__init__()
        self.pdf_num_split = pdf_num_split
        self.chunk_overlap_rate = chunk_overlap_rate

    def chunk(self, text: str) -> list[str]:
        text_len = len(text)
        chunk_size = text_len // self.pdf_num_split
        chunk_overlap = int(chunk_size * self.chunk_overlap_rate)  
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,  
            separators=["\n\n", "\n", " ", ""],  
        )
        
        return text_splitter.split_text(text)
