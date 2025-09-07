from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from text_cleaning_strategy.base import TextCleaningStrategy
from chunking_strategy.base import ChunkingStrategy
from summarizer.summarizer import SummarizerLLM
from contextual_retrieval.context_embedder import ContextEmbedderLLM
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
        summarizer: SummarizerLLM = None,
        context_embedder: ContextEmbedderLLM = None
    ):
        self.db = db
        self.embeddings = embeddings
        self.text_cleaning_strategy = text_cleaning_strategy
        self.chunking_strategy = chunking_strategy
        self.summarizer = summarizer
        self.context_embedder = context_embedder

    def get_text_cleaning_strategy_name(self) -> str:
        return self.text_cleaning_strategy.get_strategy_name()

    def get_chunking_strategy_name(self) -> str:
        return self.chunking_strategy.strategy_name

    def prepare_docs(self):
        self.db.setup_collection()

        data = []
        pdf_files = os.listdir(pdf_dir)
        len_pdf_files = len(pdf_files)
        for current_file_no, pdf_file in enumerate(pdf_files):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                print(Fore.CYAN + f"{current_file_no+1}/{len_pdf_files} 📄 Processing: {pdf_file}")

                with pymupdf.open(pdf_path) as doc:
                    title = doc.metadata.get("title", "")

                    pdf_whole_text = ""
                    for page in doc:
                        pdf_whole_text += page.get_text("text") + "\n"
                    
                    if self.summarizer:
                        summary = self.summarizer.summarize(pdf_whole_text)
                        summary = self.text_cleaning_strategy.clean_text(summary)
                    
                    if not title and pdf_whole_text.strip():
                        title = pdf_whole_text.split("\n")[0]

                    pdf_whole_text = self.text_cleaning_strategy.clean_text(pdf_whole_text)
                    chunks = self.chunking_strategy.chunk(pdf_whole_text)

                    for idx, chunk_content in enumerate(chunks):
                        contextual_content = None
                        if self.context_embedder:
                            contextual_content = self.context_embedder.embed_context(pdf_whole_text, chunk_content)
                        chunk_data = {
                            "file_name": pdf_file,
                            "title": title.strip(),
                            "chunk_id": f"{pdf_file}_chunk_{idx}",
                            "content": chunk_content,
                            "contextual_content": contextual_content
                        }
                        data.append(chunk_data)
                        self.db.store(
                            chunk=chunk_data,
                            embeddings=self.embeddings,
                            summary=summary if self.summarizer else None
                        )

        # Save JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Created {len(data)} {self.get_chunking_strategy_name()} → saved to {output_file}")