from chunking_strategy.base import ChunkingStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PdfBasedRecursivelySplitChunking(ChunkingStrategy):
    strategy_name = "pdf_based_recursively_split_chunking_v2"

    def __init__(self, chunk_overlap_rate: float, max_tokens: int):
        super().__init__()
        self.chunk_overlap_rate = chunk_overlap_rate
        self.max_tokens = max_tokens

    def chunk(self, text: str) -> list[str]:
        text_len = len(text)
        tokens = text.split()
        
        pdf_num_split = (len(tokens) // self.max_tokens) + 1
        
        chunk_size = text_len // pdf_num_split
        chunk_overlap = int(chunk_size * self.chunk_overlap_rate)  
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,  
            separators=["\n\n", "\n", " ", ""],  
        )
        
        return text_splitter.split_text(text)
