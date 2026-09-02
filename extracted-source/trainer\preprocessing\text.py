"""
Real text feature extraction for text_classification tasks (e.g. sentiment).
Free-text columns must NOT go through the tabular LabelEncoder path (that
would just assign a near-unique ID per row and fail on any unseen sentence
at inference) — this module builds an actual TF-IDF vocabulary instead.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


@dataclass
class TextPreprocessor:
    text_column: str
    target_column: str
    max_features: int = 2000  # capped for low-memory machines
    vectorizer_: Optional[TfidfVectorizer] = None
    target_encoder_: Optional[LabelEncoder] = None
    is_classification_target_: bool = True

    def fit_transform(self, df: pd.DataFrame, task: str, test_size: float = 0.2, random_seed: int = 42):
        texts = df[self.text_column].astype(str).fillna("")
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2),
                                            min_df=1, stop_words="english")
        X = self.vectorizer_.fit_transform(texts).toarray().astype(float)

        y_raw = df[self.target_column]
        self.target_encoder_ = LabelEncoder()
        y = self.target_encoder_.fit_transform(y_raw.astype(str))

        _, counts = np.unique(y, return_counts=True)
        stratify = y if counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=stratify
        )
        return {
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "n_classes": len(set(y)), "missing_report": {},
        }

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        texts = df[self.text_column].astype(str).fillna("")
        return self.vectorizer_.transform(texts).toarray().astype(float)

    def decode_target(self, y_encoded):
        return self.target_encoder_.inverse_transform(np.asarray(y_encoded).astype(int))

    def to_state(self) -> Dict[str, Any]:
        return {
            "kind": "text",
            "text_column": self.text_column,
            "target_column": self.target_column,
            "vocabulary": self.vectorizer_.vocabulary_,
            "idf": self.vectorizer_.idf_.tolist(),
            "max_features": self.max_features,
            "target_encoder_classes": self.target_encoder_.classes_.tolist(),
        }

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "TextPreprocessor":
        p = TextPreprocessor(text_column=state["text_column"], target_column=state["target_column"],
                              max_features=state["max_features"])
        vec = TfidfVectorizer(max_features=state["max_features"], ngram_range=(1, 2),
                               stop_words="english", vocabulary=state["vocabulary"])
        vec.idf_ = np.array(state["idf"])
        # sklearn needs fixed vocabulary_ + fitted internals; fit on an empty
        # doc set with the fixed vocabulary to initialize internal structures.
        vec._validate_vocabulary()
        vec.vocabulary_ = state["vocabulary"]
        p.vectorizer_ = vec
        te = LabelEncoder()
        te.classes_ = np.array(state["target_encoder_classes"])
        p.target_encoder_ = te
        return p
