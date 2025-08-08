from vector_db.vector_db import MyWeaviateDB
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style, init
import pymupdf
import os
import json
from vector_db.chunking_strategy import section_based_chunking
init(autoreset=True)


class RAG_Chatbot:
    
    def __init__(self):
        embeddings = SentenceTransformer("intfloat/multilingual-e5-base")
        self.db = MyWeaviateDB(embeddings=embeddings)
    
    def setup(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        pdf_dir = os.path.join(BASE_DIR, "vector_db/data/raw_pdfs")
        output_file = os.path.join(BASE_DIR, "vector_db/data/processed_pdfs/pdf_chunks.json")
        
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

                    if not title and text.strip():
                        title = text.split("\n")[0]

                    # Chunk with wider grouping
                    chunks = section_based_chunking(text, max_items=10)

                    for idx, chunk in enumerate(chunks):
                        data.append(
                            {
                                "file_name": pdf_file,
                                "title": title.strip(),
                                "chunk_id": f"{pdf_file}_chunk_{idx}",
                                "content": chunk,
                            }
                        )
                        self.db.store(
                            {
                                "file_name": pdf_file,
                                "title": title.strip(),
                                "chunk_id": f"{pdf_file}_chunk_{idx}",
                                "content": chunk,
                            }
                        )

        # Save JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Created {len(data)} section-based chunks → saved to {output_file}")
        
    def test_retriever(self, alpha=1, k=3):
        while True:
            query = input('\nquery: ')
            docs = self.db.search(query, alpha=alpha, k=k)
            
            for idx, relevant_doc in enumerate(docs):
                print(Fore.GREEN + f'{idx + 1}. {relevant_doc.properties["content"]}')

def main():
    chatbot = RAG_Chatbot()
    chatbot.setup()
    chatbot.test_retriever()

if __name__ == '__main__':
    main()