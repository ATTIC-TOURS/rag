from retriever import Retriever
from colorama import Fore, init
from sentence_transformers import SentenceTransformer

from vector_db.chunking_strategy import section_based_chunking

init(autoreset=True)


class RAG_Chatbot:

    def __init__(self):
        embeddings: SentenceTransformer = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )
        self.retriever = Retriever(embeddings=embeddings)

    def store_docs(self):
        self.retriever.pre_compute_docs(section_based_chunking)

    def test_retriever(self, alpha: int = 1, k: int = 3):
        color: dict[int, str] = {
            0: Fore.BLUE,
            1: Fore.GREEN,
            2: Fore.YELLOW,
            3: Fore.CYAN,
        }
        while True:
            query = input("\nquery: ")

            if query.lower() == "q":
                break

            docs = self.retriever.retrieve_relevant_docs(query=query, alpha=alpha, k=k)

            for idx, relevant_doc in enumerate(docs):
                print(
                    color[idx % len(color.values())]
                    + f'{idx + 1}. {relevant_doc.properties["content"]}'
                )


def main():
    chatbot = RAG_Chatbot()
    chatbot.store_docs()
    chatbot.test_retriever(k=10)


if __name__ == "__main__":
    main()
