from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from text_cleaning_strategy.base import TextCleaningStrategy
from chunking_strategy.base import ChunkingStrategy
from summarizer.summarizer import SummarizerLLM
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
        text_cleaning_strategy: TextCleaningStrategy,
        chunking_strategy: ChunkingStrategy,
        summarizer: SummarizerLLM = None
    ):
        self.db = db
        self.embeddings = embeddings
        self.text_cleaning_strategy = text_cleaning_strategy
        self.chunking_strategy = chunking_strategy
        self.summarizer = summarizer

    def get_text_cleaning_strategy_name(self) -> str:
        return self.text_cleaning_strategy.get_strategy_name()

    def get_chunking_strategy_name(self) -> str:
        return self.chunking_strategy.strategy_name

    def prepare_docs(self):
        self.db.setup_collection()

        data = []
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                print(Fore.CYAN + f"📄 Processing: {pdf_file}")

                with pymupdf.open(pdf_path) as doc:
                    title = doc.metadata.get("title", "")

                    text = ""
                    for page in doc:
                        text += page.get_text("text") + "\n"
                    
                    if self.summarizer:
                        summary = self.summarizer.summarize(text)
                        summary = self.text_cleaning_strategy.clean_text(summary)
                    
                    if not title and text.strip():
                        title = text.split("\n")[0]

                    text = self.text_cleaning_strategy.clean_text(text)
                    chunks = self.chunking_strategy.chunk(text)

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
                            summary=summary if self.summarizer else None
                        )

        # Save JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Created {len(data)} {self.get_chunking_strategy_name()} → saved to {output_file}")