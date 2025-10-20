from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
# import ollama


class ContextEmbedderLLM:
    def __init__(self, model_name: str = "gpt-5-mini"):
        self.model_name = model_name
        self.client = OpenAI()  # or ollama.Client()

    def embed_context(self, whole_doc: str, chunk_content: str) -> str:
        system_prompt = (
            "You are a context compression system that enhances vector embeddings by generating short "
            "contextual anchors. You must always respond with a single, concise sentence that situates a "
            "given chunk within the scope of a larger document. Strictly avoid any explanation, preface, "
            "or meta-commentary."
        )

        user_prompt = (
            "<document>\n"
            f"{whole_doc}\n"
            "</document>\n\n"
            "<chunk>\n"
            f"{chunk_content}\n"
            "</chunk>\n\n"
            "Task:\n"
            "Generate exactly one short contextual sentence (max 25 words) that clarifies how this chunk fits "
            "within the entire document, to improve embedding retrieval relevance.\n\n"
            "Rules:\n"
            "- Do NOT include the chunk in your output.\n"
            "- Do NOT explain or add commentary.\n"
            "- Do NOT say 'Here is the context', 'This chunk', or anything similar.\n"
            "- Output ONLY the pure context sentence.\n\n"
            "Return only the sentence."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        # if ollama
        # response = self.client.generate(
        #     model=self.model_name,
        #     prompt=prompt,
        #     options={"temperature": 0.0},
        # )

        context = response.choices[0].message.content.strip()

        # Build final embedding string
        return f"{context} --- {chunk_content}"

