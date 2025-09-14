import gradio as gr
from rag_pipeline.rag_pipeline import RagPipeline
from rag_pipeline.classifier.japan_visa_related_or_not.modules import (
    MyTextCleaner,
    MyEmbeddingTransformer,
)


def run_prototype(rag_pipeline: RagPipeline) -> None:
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
                {
                    "role": "assistant",
                    "content": "thinking..",
                    "name": "Attic Bot",
                    "avatar": "🤖",
                }
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
            for token in rag_pipeline.answer(message):
                bot_reply += token
                chat_history[-1]["content"] = bot_reply
                yield "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=True)
    demo.launch(share=True)


def main():
    chatbot = RagPipeline()
    chatbot.prepare_docs(from_google_drive=True)
    run_prototype(chatbot)


if __name__ == "__main__":
    main()
