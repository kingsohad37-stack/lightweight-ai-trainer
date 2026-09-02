"""
Real preprocessing for tabular data: missing-value handling, encoding,
scaling, and train/test splitting. No shortcuts that would corrupt data
silently — every transform records what it did so it can be reproduced
at inference time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class TabularPreprocessor:
    input_columns: List[str]
    target_column: Optional[str] = None
    scaler_: Optional[StandardScaler] = None
    label_encoders_: Dict[str, LabelEncoder] = field(default_factory=dict)
    target_encoder_: Optional[LabelEncoder] = None
    target_scaler_: Optional[StandardScaler] = None
    fitted_columns_: List[str] = field(default_factory=list)
    is_classification_target_: bool = False

    def fit_transform(self, df: pd.DataFrame, task: str, test_size: float = 0.2, random_seed: int = 42):
        df = df.copy()

        missing_report = {}
        for col in self.input_columns + ([self.target_column] if self.target_column else []):
            n_missing = int(df[col].isna().sum())
            if n_missing:
                missing_report[col] = n_missing
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

        X_df = df[self.input_columns].copy()
        for col in self.input_columns:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                self.label_encoders_[col] = le

        self.fitted_columns_ = list(X_df.columns)
        X = X_df.to_numpy(dtype=float)

        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        y = None
        if self.target_column:
            y_raw = df[self.target_column]
            if task in {"classification", "text_classification"} or not pd.api.types.is_numeric_dtype(y_raw):
                self.is_classification_target_ = True
                self.target_encoder_ = LabelEncoder()
                y = self.target_encoder_.fit_transform(y_raw.astype(str))
            else:
                self.is_classification_target_ = False
                y = y_raw.to_numpy(dtype=float)

        if y is not None:
            stratify = None
            if self.is_classification_target_ and len(set(y)) > 1:
                # Stratification needs every class to have at least 2 members
                # (so both train and test get at least one). Fall back to a
                # plain random split for tiny/imbalanced datasets rather than
                # crashing.
                _, counts = np.unique(y, return_counts=True)
                if counts.min() >= 2:
                    stratify = y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=stratify
            )
            return {
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "missing_report": missing_report,
                "n_classes": len(set(y)) if self.is_classification_target_ else None,
            }
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_seed)
            return {"X_train": X_train, "X_test": X_test, "missing_report": missing_report}

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply a previously-fitted transform to new data (for inference)."""
        df = df.copy()
        X_df = df[self.input_columns].copy()
        for col in self.input_columns:
            if col in self.label_encoders_:
                le = self.label_encoders_[col]
                X_df[col] = X_df[col].astype(str).map(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)
        X = X_df.to_numpy(dtype=float)
        return self.scaler_.transform(X)

    def decode_target(self, y_encoded):
        if self.target_encoder_ is not None:
            return self.target_encoder_.inverse_transform(np.asarray(y_encoded).astype(int))
        return y_encoded

    def to_state(self) -> Dict[str, Any]:
        """Serializable state for checkpointing."""
        return {
            "kind": "tabular",
            "input_columns": self.input_columns,
            "target_column": self.target_column,
            "fitted_columns": self.fitted_columns_,
            "scaler_mean": self.scaler_.mean_.tolist() if self.scaler_ is not None else None,
            "scaler_scale": self.scaler_.scale_.tolist() if self.scaler_ is not None else None,
            "label_encoder_classes": {k: v.classes_.tolist() for k, v in self.label_encoders_.items()},
            "target_encoder_classes": self.target_encoder_.classes_.tolist() if self.target_encoder_ is not None else None,
            "is_classification_target": self.is_classification_target_,
        }

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "TabularPreprocessor":
        p = TabularPreprocessor(input_columns=state["input_columns"], target_column=state["target_column"])
        p.fitted_columns_ = state["fitted_columns"]
        if state["scaler_mean"] is not None:
            sc = StandardScaler()
            sc.mean_ = np.array(state["scaler_mean"])
            sc.scale_ = np.array(state["scaler_scale"])
            sc.var_ = sc.scale_ ** 2
            sc.n_features_in_ = len(sc.mean_)
            p.scaler_ = sc
        for col, classes in state["label_encoder_classes"].items():
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            p.label_encoders_[col] = le
        if state["target_encoder_classes"] is not None:
            te = LabelEncoder()
            te.classes_ = np.array(state["target_encoder_classes"])
            p.target_encoder_ = te
        p.is_classification_target_ = state["is_classification_target"]
        return p
