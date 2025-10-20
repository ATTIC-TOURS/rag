from langchain_text_splitters import RecursiveCharacterTextSplitter


class PdfBasedCustomSplitter:
    """
    Custom text splitter for PDF documents that splits text recursively
    using a combination of separators and a maximum token limit with overlap.
    """

    def __init__(
        self,
        chunk_overlap_rate: float,
        max_tokens: int = 1000,
        separators: list[str] | None = None,
    ):
        """
        :param chunk_overlap_rate: Fraction of overlap between chunks (0.0 - 1.0)
        :param max_tokens: Approximate maximum number of tokens per chunk
        :param separators: List of separators for recursive splitting
        """
        self.chunk_overlap_rate = chunk_overlap_rate
        self.max_tokens = max_tokens
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        # Estimate number of splits based on tokens
        tokens = text.split()
        num_splits = (len(tokens) // self.max_tokens) + 1
        chunk_size = max(1, len(text) // num_splits)
        chunk_overlap = int(chunk_size * self.chunk_overlap_rate)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
        )
        return splitter.split_text(text)
