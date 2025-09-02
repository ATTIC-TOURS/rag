from retriever import Retriever
from colorama import Fore, init
from sentence_transformers import SentenceTransformer
import ollama
from retriever.prepare_docs_strategy import SectionBasedChunkPreparation
from vector_db.vector_db import MyWeaviateDB
from prompts.strategy_base import PromptStrategy
from prompts.strategy_v1 import PromptStrategyV1
import gradio as gr

init(autoreset=True)


class RAG_Chatbot:

    def __init__(self):
        self.embeddings: SentenceTransformer = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )
        self.db: MyWeaviateDB = MyWeaviateDB(
            ef_construction=300, bm25_b=0.7, bm25_k1=1.25
        )
        self.retriever = Retriever(db=self.db, embeddings=self.embeddings)

    def prepare_docs(self) -> None:
        prepareDocsStrategy = SectionBasedChunkPreparation(
            db=self.db, embeddings=self.embeddings
        )
        self.retriever.prepare_docs(prepareDocsStrategy=prepareDocsStrategy)

    def _retrieved_relevant_docs(
        self, query: str, alpha: int = 0.8, top_k: int = 3
    ) -> list[str]:
        relevant_docs = []
        for relevant_doc in self.retriever.search(query, alpha=alpha, top_k=top_k):
            relevant_docs.append(relevant_doc.properties["content"])
        return relevant_docs

    def _get_messages(self, query: str, context: list[str]) -> list[dict[str, str]]:
        promptStrategy: PromptStrategy = PromptStrategyV1()
        return promptStrategy.get_messages(query, context)

    def _generate_response(self, messages: list[dict[str, str]]):
        stream = ollama.chat(model="gemma:2b", messages=messages, stream=True)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def answer(self, query: str) -> str:
        relevant_docs = self._retrieved_relevant_docs(query, top_k=1)  # retriever
        messages = self._get_messages(query=query, context=relevant_docs)  # prompt
        return self._generate_response(messages)  # generation


def run_prototype(rag_chatbot: RAG_Chatbot) -> None:
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages", label="Japan Visa Attic Tours Chatbot")
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            # 1. Show user’s input immediately
            chat_history.append(
                {"role": "user", "content": message, "name": "You", "avatar": "👤"}
            )
            yield "", chat_history

            # 2. Show placeholder while retrieving docs
            chat_history.append(
                {"role": "assistant", "content": "⏳ Retrieving relevant documents...","name": "Attic Bot",
                "avatar": "🤖",}
            )
            yield "", chat_history

            # 3. Now stream bot reply
            bot_reply = ""
            # replace placeholder with actual streaming response
            chat_history[-1] = {
                "role": "assistant",
                "content": "",
                "name": "Attic Bot",
                "avatar": "🤖",
            }
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
