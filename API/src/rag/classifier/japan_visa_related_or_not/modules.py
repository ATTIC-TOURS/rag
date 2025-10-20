from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


class MyTextCleaner(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.apply(lambda x: x.lower())


class MyEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        texts = [f"query: {x}" for x in X.tolist()]
        return np.array(
            self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        )