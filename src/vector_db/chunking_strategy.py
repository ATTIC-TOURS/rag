import re


def section_based_chunking(text, max_items=10):
    """
    Groups related lines into bigger chunks (max_items = how many requirement items to group).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n") # cleaning
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    chunks = []
    current_chunk = ""
    item_count = 0
    
    for line in lines:
        # Start of major heading (A. PURPOSE, B. REQUIREMENTS)
        if re.match(r"^[A-Z]\.\s", line):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                item_count = 0
            current_chunk = line
        
        # Numbered item (1), (2), (3)
        elif re.match(r"^\(\d+\)", line):
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
                current_chunk = ""
                item_count = 0
            current_chunk = line
        
        else:
            # Add bullets, sub-text, or anything else
            current_chunk += "\n" + line
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks