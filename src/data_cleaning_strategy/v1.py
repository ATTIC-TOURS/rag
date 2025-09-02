from data_cleaning_strategy.base import DataCleaningStrategy
import re


class DataCleaningStrategyV1(DataCleaningStrategy):
    
    strategy_name = "cleaning_strategy_v1"
    
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