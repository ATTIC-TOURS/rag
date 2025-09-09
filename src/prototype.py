import gradio as gr
from rag_chatbot import RAG_Chatbot


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
    # chatbot.prepare_docs()
    run_prototype(chatbot)


if __name__ == "__main__":
    main()