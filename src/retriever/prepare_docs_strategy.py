from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from data_cleaning_strategy.base import DataCleaningStrategy
from chunking_strategy.base import ChunkingStrategy
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
    str_chunk_strategy = ""

    def __init__(
        self,
        db: MyWeaviateDB,
        embeddings: SentenceTransformer,
        data_cleaning_strategy: DataCleaningStrategy,
        chunking_strategy: ChunkingStrategy
    ):
        self.db = db
        self.embeddings = embeddings
        self.data_cleaning_strategy = data_cleaning_strategy
        self.chunking_strategy = chunking_strategy

    def get_data_cleaning_strategy_name(self) -> str:
        return self.data_cleaning_strategy.strategy_name

    def get_chunking_strategy_name(self) -> str:
        return self.chunking_strategy.strategy_name

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
                    chunks = self.chunking_strategy.chunk(self.data_cleaning_strategy.clean_text(text))

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