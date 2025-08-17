import pymupdf
import os
import json
from vector_db.chunking_strategy import section_based_chunking
from vector_db.vector_db import MyWeaviateDB
from colorama import Fore, init

init(autoreset=True)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version.*")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(BASE_DIR, "data/raw_pdfs")
output_file = os.path.join(BASE_DIR, "data/processed_pdfs/pdf_chunks.json")


def store_pdf_vectors(db: MyWeaviateDB, chunk: type[section_based_chunking]) -> None:
    data = []
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(Fore.CYAN + f"📄 Processing: {pdf_file}")

            '''
            It opens a PDF → gets the title from metadata if available 
            → otherwise extracts all text 
            → and if the title is missing, it uses the first line of the document as the title.
            '''
            with pymupdf.open(pdf_path) as doc:
                title = doc.metadata.get("title", "")

                text = ""
                for page in doc:
                    text += page.get_text("text") + "\n"

                if not title and text.strip():
                    title = text.split("\n")[0]

                # Chunk with wider grouping
                chunks = chunk(text, max_items=10)

                for idx, chunk_data in enumerate(chunks):
                    data.append(
                        {
                            "file_name": pdf_file,
                            "title": title.strip(),
                            "chunk_id": f"{pdf_file}_chunk_{idx}",
                            "content": chunk_data,
                        }
                    )
                    db.store(
                        {
                            "file_name": pdf_file,
                            "title": title.strip(),
                            "chunk_id": f"{pdf_file}_chunk_{idx}",
                            "content": chunk_data,
                        }
                    )

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Created {len(data)} section-based chunks → saved to {output_file}")
