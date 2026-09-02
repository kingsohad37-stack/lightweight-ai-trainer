from __future__ import annotations

import csv
import io
import json
import os
import urllib.error
import urllib.request
import zipfile
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from trainer.api import main as original

app = original.app

_root_mount = None
for route in list(app.routes):
    if getattr(route, "path", None) == "/" and hasattr(route, "app"):
        _root_mount = route
        app.router.routes.remove(route)

for route in list(app.routes):
    if getattr(route, "path", None) == "/api/generate" and "POST" in getattr(route, "methods", set()):
        app.router.routes.remove(route)


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 90) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise HTTPException(status_code=502, detail=f"AI provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach AI provider: {exc.reason}") from exc


class AIGenerateRequest(BaseModel):
    provider: str = Field(default="auto", min_length=2, max_length=30)
    model: Optional[str] = Field(default=None, max_length=120)
    prompt: str = Field(min_length=1, max_length=20_000)
    api_key: Optional[str] = Field(default=None, max_length=500)
    temperature: float = Field(default=0.7, gt=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=4096)


def _provider_config(request: AIGenerateRequest):
    provider = request.provider.lower().strip()
    if provider == "auto":
        provider = os.environ.get("AI_PROVIDER", "gemini").lower().strip()
    key = request.api_key or {
        "gemini": os.environ.get("GEMINI_API_KEY"),
        "groq": os.environ.get("GROQ_API_KEY"),
        "openai": os.environ.get("OPENAI_API_KEY"),
        "custom": os.environ.get("AI_API_KEY"),
    }.get(provider)
    if not key:
        raise HTTPException(400, f"No API key configured for provider '{provider}'. Add it in the UI or Render environment variables.")
    defaults = {
        "gemini": os.environ.get("GEMINI_MODEL", "gemini-3.7-flash"),
        "groq": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        "openai": os.environ.get("OPENAI_MODEL", "gpt-5"),
        "custom": os.environ.get("AI_MODEL", "gpt-4o-mini"),
    }
    return provider, key, request.model or defaults.get(provider, defaults["custom"])


@original.app.post("/api/ai/generate", dependencies=[Depends(original.require_api_key)])
def ai_generate(request: AIGenerateRequest):
    provider, key, model = _provider_config(request)
    if provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {"contents": [{"role": "user", "parts": [{"text": request.prompt}]}], "generationConfig": {"temperature": request.temperature, "maxOutputTokens": request.max_tokens}}
        data = _post_json(url, {"Content-Type": "application/json", "x-goog-api-key": key}, payload)
        text = "".join(part.get("text", "") for c in data.get("candidates", []) for part in c.get("content", {}).get("parts", []) if isinstance(part.get("text"), str))
        if not text:
            raise HTTPException(502, "Gemini returned no text content.")
        return {"provider": provider, "model": model, "text": text}
    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
    elif provider == "custom":
        url = os.environ.get("AI_BASE_URL", "").strip()
        if not url:
            raise HTTPException(400, "AI_BASE_URL is required for the custom OpenAI-compatible provider.")
    else:
        raise HTTPException(400, "Unsupported provider. Use gemini, groq, openai, custom, or auto.")
    payload = {"model": model, "messages": [{"role": "user", "content": request.prompt}], "temperature": request.temperature, "max_tokens": request.max_tokens}
    data = _post_json(url, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, payload)
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(502, "AI provider returned an unexpected response format.") from exc
    return {"provider": provider, "model": model, "text": text}


class DatasetGenerateRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=1000)
    columns: str = Field(min_length=1, max_length=2000)
    rows: int = Field(default=100, ge=5, le=5000)
    format: str = Field(default="csv", pattern="^(csv|jsonl|txt)$")
    api_key: Optional[str] = Field(default=None, max_length=500)
    model: Optional[str] = Field(default=None, max_length=120)


def _extract_json_array(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Gemini did not return a JSON array.")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("Generated dataset must be an array of objects.")
    return value


@original.app.post("/api/ai/dataset", dependencies=[Depends(original.require_api_key)])
@original.app.post("/api/ai/dataset/generate", dependencies=[Depends(original.require_api_key)])
def generate_dataset(request: DatasetGenerateRequest):
    key = request.api_key or os.environ.get("DATASET_GEMINI_API_KEY")
    if not key:
        raise HTTPException(400, "A Gemini API key is required for dataset generation. Add DATASET_GEMINI_API_KEY on Render or enter your own key.")
    model = request.model or os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
    columns = [c.strip() for c in request.columns.split(",") if c.strip()]
    if not columns:
        raise HTTPException(400, "Enter at least one column name.")
    prompt = f"Create exactly {request.rows} high-quality synthetic training records about: {request.topic}\n\nColumns: {', '.join(columns)}\n\nReturn ONLY a JSON array of objects. Every object must contain exactly these columns. Keep values useful for machine-learning training, varied, realistic, internally consistent, and free of markdown. Do not add explanations outside the JSON array."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7, "maxOutputTokens": min(32768, max(2048, request.rows * len(columns) * 40)), "responseMimeType": "application/json", "responseSchema": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {column: {"type": "STRING"} for column in columns}, "required": columns}}}}
    data = _post_json(url, {"Content-Type": "application/json", "x-goog-api-key": key}, payload, timeout=180)
    raw = "".join(part.get("text", "") for c in data.get("candidates", []) for part in c.get("content", {}).get("parts", []) if isinstance(part.get("text"), str))
    if not raw:
        raise HTTPException(502, "Gemini returned no dataset content.")
    try:
        records = _extract_json_array(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(502, f"Could not validate Gemini's dataset: {exc}") from exc
    records = [{column: str(row.get(column, "")) for column in columns} for row in records[:request.rows]]
    if len(records) < 5:
        raise HTTPException(502, "Gemini returned too few valid records. Try again with a smaller dataset size.")
    if request.format == "jsonl":
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n"
        filename, media_type = "ai-generated-dataset.jsonl", "application/jsonl"
    elif request.format == "txt":
        content = "\n\n".join("\n".join(f"{column}: {row[column]}" for column in columns) for row in records) + "\n"
        filename, media_type = "ai-generated-dataset.txt", "text/plain"
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader(); writer.writerows(records)
        content = output.getvalue()
        filename, media_type = "ai-generated-dataset.csv", "text/csv"
    return {"filename": filename, "format": request.format, "rows": len(records), "columns": columns, "content": content, "media_type": media_type}


@original.app.post("/api/generate", dependencies=[Depends(original.require_api_key)])
def generate_text(request: original.GenerateRequest):
    from trainer.algorithms.transformer import TinyTransformer
    from trainer.tokenizers.char_tokenizer import CharTokenizer
    from trainer.training.checkpoint import CheckpointManager
    try:
        name = original._safe_name(request.experiment)
        manager = CheckpointManager(str(original.MODEL_ROOT), name)
        checkpoint = manager.load_epoch(request.epoch) if request.epoch else manager.load_latest()
        if not checkpoint:
            raise HTTPException(404, "No trained checkpoint found for this experiment.")
        metadata = checkpoint.get("metadata", {})
        algorithm = metadata.get("algorithm")
        if algorithm != "tiny_transformer":
            raise HTTPException(400, f"Experiment '{name}' is a {algorithm or 'non-text'} model. Step 5 requires a language-model training run (task=language_modeling, algorithm=tiny_transformer). Your classification/regression model is still valid for prediction.")
        transformer_state = checkpoint.get("model_state", {}).get("transformer_state")
        tokenizer_state = checkpoint.get("preprocessor_state")
        if not transformer_state or not tokenizer_state:
            raise HTTPException(400, "This language-model checkpoint is incomplete: transformer/tokenizer state is missing.")
        model = TinyTransformer.from_state(transformer_state)
        tokenizer = CharTokenizer.from_state(tokenizer_state)
        prompt_ids = tokenizer.encode(request.prompt)
        generated_ids = model.generate(prompt_ids, request.max_new_tokens, request.temperature)
        return {"experiment": name, "text": tokenizer.decode(generated_ids[len(prompt_ids):]), "epoch": metadata.get("epoch")}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {exc}") from exc


class ChatMessage(BaseModel):
    role: str = Field(min_length=1, max_length=20)
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    experiment: str = Field(min_length=1, max_length=120)
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    max_new_tokens: int = Field(default=160, ge=1, le=1024)
    temperature: float = Field(default=0.8, gt=0, le=2)
    epoch: Optional[int] = Field(default=None, ge=0)


def _chat_prompt(messages: list[ChatMessage]) -> str:
    parts = []
    for message in messages:
        role = message.role.lower().strip()
        label = "System" if role == "system" else "Assistant" if role == "assistant" else "User"
        parts.append(f"{label}: {message.content.strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


@original.app.post("/api/chat", dependencies=[Depends(original.require_api_key)])
def chat(request: ChatRequest):
    from trainer.algorithms.transformer import TinyTransformer
    from trainer.tokenizers.char_tokenizer import CharTokenizer
    from trainer.training.checkpoint import CheckpointManager
    try:
        name = original._safe_name(request.experiment)
        manager = CheckpointManager(str(original.MODEL_ROOT), name)
        checkpoint = manager.load_epoch(request.epoch) if request.epoch else manager.load_latest()
        if not checkpoint:
            raise HTTPException(404, "No trained checkpoint found for this experiment.")
        metadata = checkpoint.get("metadata", {})
        if metadata.get("algorithm") != "tiny_transformer":
            raise HTTPException(400, f"Experiment '{name}' is not a language model. Train it with task=language_modeling first.")
        transformer_state = checkpoint.get("model_state", {}).get("transformer_state")
        tokenizer_state = checkpoint.get("preprocessor_state")
        if not transformer_state or not tokenizer_state:
            raise HTTPException(400, "This language-model checkpoint is incomplete: transformer/tokenizer state is missing.")
        model = TinyTransformer.from_state(transformer_state)
        tokenizer = CharTokenizer.from_state(tokenizer_state)
        prompt = _chat_prompt(request.messages)
        prompt_ids = tokenizer.encode(prompt)
        generated_ids = model.generate(prompt_ids, request.max_new_tokens, request.temperature)
        text = tokenizer.decode(generated_ids[len(prompt_ids):])
        for marker in ("\nUser:", "\nSystem:", "\nAssistant:"):
            text = text.split(marker, 1)[0]
        return {"experiment": name, "text": text.strip(), "messages": [m.model_dump() for m in request.messages], "epoch": metadata.get("epoch")}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Chat generation failed: {exc}") from exc


if _root_mount is not None:
    app.router.routes.append(_root_mount)
