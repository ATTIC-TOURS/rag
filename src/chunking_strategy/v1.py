from chunking_strategy.base import ChunkingStrategy
import re


class ChunkingStrategyV1(ChunkingStrategy):
    strategy_name = "section_based_chunking"

    def chunk(self, text: str) -> list[str]:
        max_items = 10

        text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalize line breaks
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        chunks: list[str] = []
        current_chunk: str = ""
        item_count: int = 0

        for line in lines:
            # Major heading (A. PURPOSE, B. REQUIREMENTS)
            if re.match(r"^[a-zA-Z]\.\s", line):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
                item_count = 0

            # Numbered item (1), (2), (3)
            elif (
                re.match(r"^\(\d+\)", line)
                or re.match(r"^\d+\.", line)
                or re.match(r"^・", line)
            ):
                if item_count >= max_items:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    item_count = 0
                current_chunk += "\n" + line
                item_count += 1

            # Special headings 【ADDITIONAL REQUIREMENTS】
            elif line.startswith("【") and line.endswith("】"):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
                item_count = 0

            else:
                current_chunk += "\n" + line

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
