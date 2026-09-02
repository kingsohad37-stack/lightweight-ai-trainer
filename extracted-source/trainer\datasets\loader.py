"""
Real dataset loading. No synthetic stand-ins unless explicitly requested
via the demo dataset generators. Supports streaming/chunked reads for CSV
so large files don't have to be loaded fully into RAM.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from typing import Iterator, List, Dict, Any

import pandas as pd


class DatasetLoadError(Exception):
    pass


def detect_format(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        return "text_folder"
    ext = p.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".txt":
        return "txt"
    if ext == ".tsv":
        return "tsv"
    raise DatasetLoadError(f"Unsupported file extension '{ext}' for {path}")


def load_csv(path: str, chunksize: int | None = None):
    """
    Loads a CSV. If chunksize is given, returns an iterator of DataFrame
    chunks (streaming) instead of a single DataFrame — used for large files
    on low-RAM machines.
    """
    if not os.path.exists(path):
        raise DatasetLoadError(f"File not found: {path}")
    try:
        if chunksize:
            return pd.read_csv(path, chunksize=chunksize)
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DatasetLoadError(f"CSV file is empty: {path}")
    except pd.errors.ParserError as e:
        raise DatasetLoadError(f"Could not parse CSV ({path}): {e}")


def load_json(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise DatasetLoadError(f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetLoadError(f"Invalid JSON in {path}: {e}")
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    raise DatasetLoadError(f"Unsupported JSON structure in {path}")


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Streams a JSONL file line-by-line (no full-file load)."""
    if not os.path.exists(path):
        raise DatasetLoadError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetLoadError(f"Invalid JSON on line {lineno} of {path}: {e}")


def load_jsonl(path: str) -> pd.DataFrame:
    return pd.DataFrame(list(iter_jsonl(path)))


def load_txt(path: str) -> List[str]:
    """Loads a plain text file as a list of non-empty lines (for LM/text tasks)."""
    if not os.path.exists(path):
        raise DatasetLoadError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def load_text_folder(path: str) -> List[str]:
    """Loads every .txt file in a folder (non-recursive) into a list of docs."""
    p = Path(path)
    if not p.is_dir():
        raise DatasetLoadError(f"Not a directory: {path}")
    docs = []
    files = sorted(p.glob("*.txt"))
    if not files:
        raise DatasetLoadError(f"No .txt files found in {path}")
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            docs.append(fh.read())
    return docs


def load(path: str, chunksize: int | None = None):
    """
    Generic entrypoint. Returns either a pd.DataFrame (csv/json/jsonl/tsv)
    or a List[str] (txt/text_folder), or a chunk-iterator (csv streaming).
    """
    fmt = detect_format(path)
    if fmt == "csv":
        return load_csv(path, chunksize=chunksize)
    if fmt == "tsv":
        if not os.path.exists(path):
            raise DatasetLoadError(f"File not found: {path}")
        return pd.read_csv(path, sep="\t")
    if fmt == "json":
        return load_json(path)
    if fmt == "jsonl":
        return load_jsonl(path)
    if fmt == "txt":
        return load_txt(path)
    if fmt == "text_folder":
        return load_text_folder(path)
    raise DatasetLoadError(f"Unhandled format: {fmt}")
