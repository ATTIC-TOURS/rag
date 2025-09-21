from ..vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from ..text_cleaning_strategy.base import TextCleaningStrategy
from ..chunking_strategy.base import ChunkingStrategy
from ..context_augment.context_embedder import ContextEmbedderLLM
from llama_index.readers.google import GoogleDriveReader
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
        context_embedder: ContextEmbedderLLM | None = None,
    ):
        self.db = db
        self.embeddings = embeddings
        self.text_cleaning_strategy = text_cleaning_strategy
        self.chunking_strategy = chunking_strategy
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
                print(
                    Fore.CYAN
                    + f"{current_file_no+1}/{len_pdf_files} 📄 Processing: {pdf_file}"
                )

                with pymupdf.open(pdf_path) as doc:
                    title = doc.metadata.get("title", "")

                    pdf_whole_text = ""
                    for page in doc:
                        pdf_whole_text += page.get_text("text") + "\n"

                    if not title and pdf_whole_text.strip():
                        title = pdf_whole_text.split("\n")[0]

                    pdf_whole_text = self.text_cleaning_strategy.clean_text(
                        pdf_whole_text
                    )
                    chunks = self.chunking_strategy.chunk(pdf_whole_text)

                    for idx, chunk_content in enumerate(chunks):
                        contextual_content = None
                        if self.context_embedder:
                            contextual_content = self.context_embedder.embed_context(
                                pdf_whole_text, chunk_content
                            )
                        chunk_data = {
                            "file_name": pdf_file,
                            "title": title.strip(),
                            "chunk_id": f"{pdf_file}_chunk_{idx}",
                            "content": chunk_content,
                            "contextual_content": contextual_content,
                        }
                        data.append(chunk_data)
                        self.db.store(
                            chunk=chunk_data,
                            embeddings=self.embeddings,
                        )

        # Save JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(
            f"✅ Created {len(data)} {self.get_chunking_strategy_name()} → saved to {output_file}"
        )

    def prepare_docs_from_google_drive(self):
        self.db.setup_collection()
        with open(os.path.join(BASE_DIR, "service_account_key.json")) as f:
            service_account_info = json.load(f)
        reader = GoogleDriveReader(
            folder_id="1qq125mDOBJ0sNJLh7znrFfFzaQOm7L-m",
            service_account_key=service_account_info,
        )
        data = []
        documents = reader.load_data()
        len_documents = len(documents)
        for current_file_no, document in enumerate(documents):
            if document.metadata['mime type'] == 'application/pdf':
                file_name = document.metadata['file path'].split('/')[-1]
                print(
                    Fore.CYAN
                    + f"{current_file_no+1}/{len_documents} 📄 Processing: {file_name}"
                )

                file_text = document.text_resource.text

                if file_text.strip():
                    file_title = file_text.split("\n")[0]

                    file_text = self.text_cleaning_strategy.clean_text(
                        file_text
                    )
                    chunks = self.chunking_strategy.chunk(file_text)

                    for idx, chunk_content in enumerate(chunks):
                        contextual_content = None
                        if self.context_embedder:
                            contextual_content = self.context_embedder.embed_context(
                                file_text, chunk_content
                            )
                        chunk_data = {
                            "file_name": file_name,
                            "title": file_title.strip(),
                            "chunk_id": f"{file_name}_chunk_{idx}",
                            "content": chunk_content,
                            "contextual_content": contextual_content,
                        }
                        data.append(chunk_data)
                        self.db.store(
                            chunk=chunk_data,
                            embeddings=self.embeddings,
                        )

        # Save JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(
            f"✅ Created {len(data)} {self.get_chunking_strategy_name()} → saved to {output_file}"
        )
