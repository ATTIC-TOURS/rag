from retriever import Retriever
from colorama import Fore, init
from sentence_transformers import SentenceTransformer
import ollama
from vector_db.chunking_strategy import section_based_chunking
import gradio as gr

init(autoreset=True)


class RAG_Chatbot:

    def __init__(self):
        embeddings: SentenceTransformer = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )
        self.retriever = Retriever(embeddings=embeddings)

    def prepare_relevant_docs(self) -> None:
        self.retriever.pre_compute_docs(section_based_chunking)

    def answer(self, query: str) -> str:
        relevant_docs = self.retriever.retrieve_relevant_docs(query, alpha=0.8, k=3)
        context = ""
        for relevant_doc in relevant_docs:
            context += relevant_doc.properties["content"]
        response = ollama.chat(
            model="gemma:2b",
            messages=[
                {"role": "system", "content": f"Use the following context to answer the user query:\n{context}\nYou have to answer it concise and precise."},
                {"role": "user", "content": f"user query: {query}"}
            ],
        )
        return response["message"]["content"]

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

def run_prototype(rag_chatbot: RAG_Chatbot) -> None:
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            bot_message = rag_chatbot.answer(message)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    demo.launch(share=True)
    
def main():
    chatbot = RAG_Chatbot()
    chatbot.prepare_relevant_docs()
    run_prototype(chatbot)


if __name__ == "__main__":
    main()
