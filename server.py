from __future__ import annotations

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

# Remove the original root static mount temporarily so our API routes stay
# ahead of it in Starlette's route order. Re-add it after the extensions.
_root_mount = None
for route in list(app.routes):
    if getattr(route, "path", None) == "/" and hasattr(route, "app"):
        _root_mount = route
        app.router.routes.remove(route)

# Replace the original /api/generate route with a friendlier version that
# explains when a classification/regression model cannot generate text.
for route in list(app.routes):
    if getattr(route, "path", None) == "/api/generate" and "POST" in getattr(route, "methods", set()):
        app.router.routes.remove(route)


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 90) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
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

    key = request.api_key
    if not key:
        key = {
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
        payload = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": {"temperature": request.temperature, "maxOutputTokens": request.max_tokens},
        }
        data = _post_json(url, {"Content-Type": "application/json", "x-goog-api-key": key}, payload)
        text = ""
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    text += part["text"]
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

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    data = _post_json(url, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, payload)
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(502, "AI provider returned an unexpected response format.") from exc
    return {"provider": provider, "model": model, "text": text}


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
        algorithm = checkpoint["metadata"].get("algorithm")
        if algorithm != "tiny_transformer":
            raise HTTPException(400, f"Experiment '{name}' uses {algorithm or 'a non-text model'}. Generate text requires a tiny_transformer language-model experiment.")
        transformer_state = checkpoint.get("model_state", {}).get("transformer_state")
        tokenizer_state = checkpoint.get("preprocessor_state")
        if not transformer_state or not tokenizer_state:
            raise HTTPException(400, "This language-model checkpoint is missing transformer/tokenizer state.")
        model = TinyTransformer.from_state(transformer_state)
        tokenizer = CharTokenizer.from_state(tokenizer_state)
        text = tokenizer.decode(model.generate(tokenizer.encode(request.prompt), request.max_new_tokens, request.temperature))
        return {"experiment": name, "text": text}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@original.app.get("/api/experiments/{experiment}/download", dependencies=[Depends(original.require_api_key)])
def download_experiment(experiment: str):
    from trainer.training.checkpoint import CheckpointManager

    name = original._safe_name(experiment)
    manager = CheckpointManager(str(original.MODEL_ROOT), name)
    latest = manager.load_latest()
    if not latest:
        raise HTTPException(404, "No trained model found for this experiment.")

    model_dir = (original.MODEL_ROOT / name).resolve()
    if original.MODEL_ROOT.resolve() not in model_dir.parents:
        raise HTTPException(400, "Invalid experiment path.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in model_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(model_dir))
        manifest = {
            "experiment": name,
            "algorithm": latest["metadata"].get("algorithm"),
            "epoch": latest["metadata"].get("epoch"),
            "metrics": latest["metadata"].get("metrics", {}),
            "config": latest["metadata"].get("config", {}),
        }
        archive.writestr("MODEL_MANIFEST.json", json.dumps(manifest, indent=2))
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{name}-trained-model.zip"'}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


if _root_mount is not None:
    app.router.routes.append(_root_mount)
