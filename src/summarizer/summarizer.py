from ollama import chat


class SummarizerLLM:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def summarize(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarizer. Output ONLY the summary text. "
                    "No introductions, no phrases like 'Here is the summary'. "
                    "If you understand, just output the summary without any extra words."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize the following text:\n\n{text}\n\nYour output should be just the summarized text, nothing else.",
            },
        ]

        stream = chat(
            self.model_name, messages=messages, stream=True, options={"temperature": 0.0}
        )

        summary = ""
        for chunk in stream:
            summary += chunk.get("message", {}).get("content", "")
        summary = summary.replace(
            "Sure, here is the summary you requested:", ""
        ).strip()
        return summary