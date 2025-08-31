from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
import re
import pymupdf
import os
import json
from colorama import Fore, init

init(autoreset=True)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version.*")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(BASE_DIR, "data/raw_pdfs")
output_file = os.path.join(BASE_DIR, "data/processed_pdfs/pdf_chunks.json")


class PrepareDocsStrategy:
    str_clean_strategy = ""
    str_chunk_strategy = ""

    def __init__(self, db: MyWeaviateDB, embeddings: SentenceTransformer):
        self.db = db
        self.embeddings = embeddings

    def _clean(self, text: str) -> str:
        return text

    def _chunk(self) -> list[str]:
        return []

    def prepare_docs(self):
        self.db.setup_collection()

        data = []
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                print(Fore.CYAN + f"📄 Processing: {pdf_file}")

                """
                It opens a PDF → gets the title from metadata if available 
                → otherwise extracts all text 
                → and if the title is missing, it uses the first line of the document as the title.
                """
                with pymupdf.open(pdf_path) as doc:
                    title = doc.metadata.get("title", "")

                    text = ""
                    for page in doc:
                        text += page.get_text("text") + "\n"

                    if not title and text.strip():
                        title = text.split("\n")[0]

                    # Chunk with wider grouping
                    chunks = self._chunk(self._clean(text))

                    for idx, chunk_data in enumerate(chunks):
                        temp_data = {
                            "file_name": pdf_file,
                            "title": title.strip(),
                            "chunk_id": f"{pdf_file}_chunk_{idx}",
                            "content": chunk_data,
                        }
                        data.append(temp_data)
                        self.db.store(
                            chunk=temp_data,
                            embeddings=self.embeddings,
                        )

        # Save JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Created {len(data)} section-based chunks → saved to {output_file}")


class SectionBasedChunkPreparation(PrepareDocsStrategy):

    def __init__(self, db: MyWeaviateDB, embeddings: SentenceTransformer):
        super().__init__(db, embeddings)

    def _clean(self, text: str) -> str:
        """
        1. lower the letter
        2. remove urls
        3. remove unnecessary spacing
        (brackets and numbering are kept for chunking)
        """
        self.str_clean_strategy = "cleaning_strategy_v2"

        text = text.lower()
        text = re.sub(r"(https?://\S+|www\.\S+)", "", text)  # remove URLs
        text = re.sub(r"\s+", " ", text).strip()  # normalize spacing
        return text


    def _chunk(self, text: str) -> list[str]:
        self.str_chunk_strategy = "section_based_chunking"
        """
        Groups related lines into bigger chunks (max_items = how many requirement items to group).
        """

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
