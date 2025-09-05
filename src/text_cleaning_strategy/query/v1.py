from text_cleaning_strategy.base import TextCleaningStrategy
import re


class QueryCleaningStrategyV1:

    def get_strategy_name(self) -> str:
        return "query_cleaning_strategy_v1"

    def clean_text(self, text):
        """Strategy
        1. lower the letter
        2. remove punctuation marks
        3. remove "po" and "opo"
        4. remove unnecessary spacing
        """

        text = text.lower()
        text = re.sub(r"[.!?]", "", text)  # punctuation marks
        tokens = text.split()
        tokens = [token for token in tokens if token != "po" or token != "opo"]
        text = " ".join(tokens)
        text = re.sub(r"\s+", " ", text).strip()  # normalize spacing
        return text
