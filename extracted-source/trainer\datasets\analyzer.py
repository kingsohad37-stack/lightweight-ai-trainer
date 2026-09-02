"""
Real dataset analysis. Every number below is computed directly from the
actual loaded data — nothing here is fabricated or templated.
"""
from __future__ import annotations
import os
import re
from typing import List, Union

import pandas as pd
import numpy as np


def _is_probably_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def analyze_tabular(df: pd.DataFrame, path: str | None = None) -> dict:
    """Real column-by-column analysis of a tabular (CSV/JSON/JSONL) dataset."""
    n_rows, n_cols = df.shape
    file_size = os.path.getsize(path) if path and os.path.exists(path) else None

    columns = {}
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        col_info = {
            "dtype": str(series.dtype),
            "missing_values": missing,
            "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0.0,
            "unique_values": int(series.nunique(dropna=True)),
        }
        if _is_probably_numeric(series):
            desc = series.describe()
            col_info.update({
                "kind": "numeric",
                "min": float(desc.get("min", float("nan"))),
                "max": float(desc.get("max", float("nan"))),
                "mean": float(desc.get("mean", float("nan"))),
                "std": float(desc.get("std", float("nan"))),
            })
        else:
            n_unique = series.nunique(dropna=True)
            col_info["kind"] = "categorical" if n_unique <= max(20, n_rows * 0.05) else "text"
            if col_info["kind"] == "categorical":
                vc = series.value_counts(dropna=True).to_dict()
                col_info["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            else:
                lengths = series.dropna().astype(str).str.split().apply(len)
                col_info["avg_word_count"] = float(lengths.mean()) if len(lengths) else 0.0
                col_info["approx_vocab_size"] = int(
                    pd.Series(" ".join(series.dropna().astype(str)).lower().split()).nunique()
                ) if len(series.dropna()) else 0
        columns[col] = col_info

    duplicate_rows = int(df.duplicated().sum())

    return {
        "n_samples": int(n_rows),
        "n_columns": int(n_cols),
        "file_size_bytes": file_size,
        "columns": columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / n_rows * 100, 2) if n_rows else 0.0,
    }


_WORD_RE = re.compile(r"\b\w+\b")


def analyze_text(docs: List[str], path: str | None = None) -> dict:
    """Real statistics over a list of text documents/lines (for LM/text tasks)."""
    n_docs = len(docs)
    file_size = os.path.getsize(path) if path and os.path.exists(path) else None

    total_chars = sum(len(d) for d in docs)
    tokens_per_doc = [len(_WORD_RE.findall(d)) for d in docs]
    total_tokens = int(sum(tokens_per_doc))
    vocab = set()
    for d in docs:
        vocab.update(w.lower() for w in _WORD_RE.findall(d))

    return {
        "n_documents": n_docs,
        "file_size_bytes": file_size,
        "total_characters": total_chars,
        "approx_total_tokens": total_tokens,
        "avg_tokens_per_doc": float(np.mean(tokens_per_doc)) if tokens_per_doc else 0.0,
        "vocab_size": len(vocab),
        "duplicate_documents": int(n_docs - len(set(docs))),
    }


def suggest_split(n_samples: int, test_size: float = 0.2) -> dict:
    n_test = int(round(n_samples * test_size))
    n_train = n_samples - n_test
    return {"n_train": n_train, "n_test": n_test, "test_size": test_size}
