import ollama


class ContextEmbedderLLM:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_context(self, whole_doc: str, chunk_content: str) -> str:
        prompt = (
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

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": 0.0},
        )

        context = (
            response.response.replace("Sure, here's the context you requested:", "")
            .replace("Sure, here is the context you requested:", "")
            .replace("Sure, here's the context:", "")
            .replace("Sure, here is the context:", "")
            .strip()
        )

        return f"{context} {chunk_content}"
