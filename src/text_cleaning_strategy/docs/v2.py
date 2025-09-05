from text_cleaning_strategy.base import TextCleaningStrategy
import re


class DocsCleaningStrategyV2(TextCleaningStrategy):
    
    def get_strategy_name(self) -> str:
        return "docs_cleaning_strategy_v2"
    
    def clean_text(self, text):
        """Strategy
        1. lower the letter
        2. remove brackets
        3. remove multiple dots
        4. remove unnecessary spacing
        """

        text = text.lower()
        text = re.sub(r"[()\[\]{}]", "", text)  # remove brackets
        text = re.sub(r"\.{2,}", ".", text) # remove multiple dots
        text = re.sub(r"\s+", " ", text).strip()  # normalize spacing
        return text