from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import server

MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", str(Path(__file__).resolve().parent / ".local-model"))
_MODEL = None
_TOKENIZER = None


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        if not os.path.isdir(MODEL_DIR):
            raise server.HTTPException(503, "The local dataset model is not installed on this server yet. Rebuild the service and try again.")
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        _MODEL.eval()
        if _TOKENIZER.pad_token_id is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
    return _MODEL, _TOKENIZER


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Local model did not return a JSON array.")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Generated dataset must be an array of objects.")
    return value


def _generate_batch(model, tokenizer, topic: str, columns: list[str], rows: int) -> list[dict[str, Any]]:
    prompt = (
        "You are a synthetic training-data generator. "
        f"Create exactly {rows} realistic, varied records about {topic!r}. "
        f"Every record must contain exactly these fields: {', '.join(columns)}. "
        "Return ONLY a valid JSON array. No markdown, comments, or explanation. "
        "Use useful, internally consistent values suitable for machine-learning training."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=min(4096, max(512, rows * max(20, len(columns) * 18))),
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return _extract_json_array(tokenizer.decode(generated, skip_special_tokens=True))


def generate_dataset(request: server.DatasetGenerateRequest):
    columns = [c.strip() for c in request.columns.split(",") if c.strip()]
    if not columns:
        raise server.HTTPException(400, "Enter at least one column name.")
    model, tokenizer = _load_model()
    records: list[dict[str, str]] = []
    batch_size = 20
    attempts = 0
    while len(records) < request.rows and attempts < max(8, (request.rows // batch_size) * 3 + 3):
        attempts += 1
        wanted = min(batch_size, request.rows - len(records))
        try:
            batch = _generate_batch(model, tokenizer, request.topic, columns, wanted)
        except Exception:
            continue
        for row in batch:
            if not isinstance(row, dict):
                continue
            normalized = {column: str(row.get(column, "")).strip() for column in columns}
            if any(normalized.values()):
                records.append(normalized)
                if len(records) >= request.rows:
                    break
    if len(records) < request.rows:
        raise server.HTTPException(502, f"The local dataset model produced {len(records)} of {request.rows} requested rows. Try fewer rows or simpler columns.")
    records = records[: request.rows]
    if request.format == "jsonl":
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
        filename, media_type = "ai-generated-dataset.jsonl", "application/jsonl"
    elif request.format == "txt":
        content = "\n\n".join("\n".join(f"{column}: {row[column]}" for column in columns) for row in records) + "\n"
        filename, media_type = "ai-generated-dataset.txt", "text/plain"
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
        content = output.getvalue()
        filename, media_type = "ai-generated-dataset.csv", "text/csv"
    return {
        "filename": filename, "format": request.format, "rows": len(records), "columns": columns,
        "content": content, "media_type": media_type, "provider": "local",
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    }
