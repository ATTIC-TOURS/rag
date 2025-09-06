from ollama import chat


class ContextEmbedderLLM:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_context(self, whole_doc: str, chunk_content: str) -> str:
        user_content = (
            "<document>"
            f"{whole_doc}"
            "</document>"
            "Here is the chunk we want to situate within the whole document"
            "<chunk>"
            f"{chunk_content} "
            "</chunk> "
            "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk."
            "Answer only with the succinct context and nothing else. "
        )

        stream = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": user_content}],
            stream=True,
            options={"temperature": 0.0},
        )

        context = ""
        for chunk in stream:
            context += chunk.get("message", {}).get("content", "")
        context = context.replace("Sure, here's the context you requested:", "").strip()
        
        return f"{context} {chunk_content}"
