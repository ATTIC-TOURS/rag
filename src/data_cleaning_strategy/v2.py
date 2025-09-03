from data_cleaning_strategy.base import DataCleaningStrategy
import re


class DataCleaningStrategyV2(DataCleaningStrategy):
    
    strategy_name = "cleaning_strategy_v2"
    
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