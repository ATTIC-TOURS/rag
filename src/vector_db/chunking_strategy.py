import re
from typing import List

def section_based_chunking(text: str, max_items: int = 10) -> List[str]:
    """
    Groups related lines into bigger chunks (max_items = how many requirement items to group).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalize line breaks
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    chunks: List[str] = []
    current_chunk: str = ""
    item_count: int = 0

    for line in lines:
        # Major heading (A. PURPOSE, B. REQUIREMENTS)
        if re.match(r"^[A-Z]\.\s", line):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line
            item_count = 0

        # Numbered item (1), (2), (3)
        elif re.match(r"^\(\d+\)", line) or re.match(r"^\d+\.", line) or re.match(r"^・", line):
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
