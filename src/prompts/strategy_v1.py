from prompts.strategy_base import PromptStrategy


class PromptStrategyV1(PromptStrategy):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "PromptStrategyV1"

    def get_messages(self, query: str, context: list[str]) -> list[dict[str, str]]:
        system_content = f"""
                          You are assistant in Japan Visa related in Attic Tours Company.
                          The applicants are seeking for information about requirements.
                          Instruction\n
                          respond to the user query using the context if asking you a question.
                          Be aware of the context is not always Japan Visa, 
                          so you have to think it carefully whether you use the context or not.
                          You should be helpful.
                          The response must be restrictly related in Japan Visa.
                          You have to respect them.
                          """

        user_content = f"""
                        user query: {query}\n
                        context: {" ".join(context)}\n
                        """
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": user_content},
        ]
        return messages
