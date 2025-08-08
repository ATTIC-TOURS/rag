from vector_db.vector_db import MyWeaviateDB
from vector_db.store_vectors import store_pdf_vectors
from sentence_transformers import SentenceTransformer
from colorama import Fore, init

init(autoreset=True)


class RAG_Chatbot:

    def __init__(self):
        embeddings: SentenceTransformer = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )
        self.db: MyWeaviateDB = MyWeaviateDB(embeddings=embeddings)

    def setup(self):
        store_pdf_vectors(self.db)

    def test_retriever(self, alpha: int = 1, k: int = 3):
        while True:
            query = input("\nquery: ")
            docs = self.db.search(query, alpha=alpha, k=k)

            for idx, relevant_doc in enumerate(docs):
                print(Fore.GREEN + f'{idx + 1}. {relevant_doc.properties["content"]}')


def main():
    chatbot = RAG_Chatbot()
    chatbot.setup()
    chatbot.test_retriever()


if __name__ == "__main__":
    main()
