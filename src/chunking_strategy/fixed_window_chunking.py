from chunking_strategy.base import ChunkingStrategy


class FixedWindowChunking(ChunkingStrategy):
    strategy_name = "fixed_window_chunking"

    def __init__(self, window_size: int = 100, overlap_size: int = 0):
        super().__init__()
        self.window_size = window_size
        self.overlap_size = overlap_size

    def chunk(self, text: str) -> list[str]:
        tokens = text.split()

        chunks = []
        starting_idx = 0
        while len(tokens) > starting_idx:
            chunk = " ".join(tokens[starting_idx : starting_idx + self.window_size])
            chunks.append(chunk)
            starting_idx = starting_idx + self.window_size - self.overlap_size
        return chunks
