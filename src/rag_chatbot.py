from retriever import Retriever
from colorama import Fore, init
from sentence_transformers import SentenceTransformer
import ollama
from retriever.prepare_docs_strategy import SectionBasedChunkPreparation
from vector_db.vector_db import MyWeaviateDB
import gradio as gr

init(autoreset=True)


class RAG_Chatbot:

    def __init__(self):
        self.embeddings: SentenceTransformer = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )
        self.db: MyWeaviateDB = MyWeaviateDB(
            ef_construction=300,
            bm25_b=0.7,
            bm25_k1=1.25
        )
        self.retriever = Retriever(db=self.db, embeddings=self.embeddings)

    def prepare_docs(self) -> None:
        prepareDocsStrategy = SectionBasedChunkPreparation(db=self.db, embeddings=self.embeddings)
        self.retriever.prepare_docs(prepareDocsStrategy=prepareDocsStrategy)
    
    def _retrieved_relevant_docs(self, query: str, alpha: int = 0.8, top_k: int = 3) -> str:
        relevant_docs = self.retriever.search(query, alpha=alpha, top_k=top_k)
        context = ""
        for relevant_doc in relevant_docs:
            context += relevant_doc.properties["content"]
        return context
    
    def _generate_response(self, query: str, context: str):
        stream = ollama.chat(
            model="gemma:2b",
            messages=[
                {"role": "system", "content": f"Use the following context to answer the user query:\n{context}\nYou have to answer it concise and precise."},
                {"role": "user", "content": f"user query: {query}"}
            ],
            stream=True
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def answer(self, query: str) -> str:
        context = self._retrieved_relevant_docs(query)
        return self._generate_response(query, context)
         

def run_prototype(rag_chatbot: RAG_Chatbot) -> None:
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            # 1. Show user’s input immediately
            chat_history.append({"role": "user", "content": message})
            yield "", chat_history

            # 2. Show placeholder while retrieving docs
            chat_history.append({"role": "assistant", "content": "⏳ Retrieving relevant documents..."})
            yield "", chat_history

            # 3. Now stream bot reply
            bot_reply = ""
            # replace placeholder with actual streaming response
            chat_history[-1] = {"role": "assistant", "content": ""}
            for token in rag_chatbot.answer(message):
                bot_reply += token
                chat_history[-1]["content"] = bot_reply
                yield "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)
    demo.launch(share=True)


    
def main():
    chatbot = RAG_Chatbot()
    chatbot.prepare_docs()
    run_prototype(chatbot)


if __name__ == "__main__":
    main()
