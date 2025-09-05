from text_cleaning_strategy.base import TextCleaningStrategy
import re


class DocsCleaningStrategyV1(TextCleaningStrategy):
    
    def get_strategy_name(self) -> str:
        return "docs_cleaning_strategy_v1"
    
    def clean_text(self, text):
        """Strategy
        1. lower the letter
        2. remove urls
        3. remove unnecessary spacing
        """

        text = text.lower()
        text = re.sub(r"(https?://\S+|www\.\S+)", "", text)  # remove URLs
        text = re.sub(r"\s+", " ", text).strip()  # normalize spacing
        return text