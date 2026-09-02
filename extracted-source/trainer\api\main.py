"""Safe, single-instance web API for the Lightweight AI Trainer."""
from __future__ import annotations

import hmac
import logging
import math
import os
import re
import threading
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("TRAINER_DATA_DIR", APP_ROOT / "data")).resolve()
UPLOAD_ROOT, MODEL_ROOT, EXPERIMENT_ROOT = DATA_ROOT / "uploads", DATA_ROOT / "models", DATA_ROOT / "experiments"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
MAX_ACTIVE_JOBS = int(os.environ.get("MAX_ACTIVE_JOBS", 1))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 200))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 512))
MAX_HIDDEN_UNITS = int(os.environ.get("MAX_HIDDEN_UNITS", 512))
API_KEY = os.environ.get("TRAINER_API_KEY")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
LOG = logging.getLogger("trainer.api")

for path in (UPLOAD_ROOT, MODEL_ROOT, EXPERIMENT_ROOT):
    path.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Lightweight AI Trainer", version="1.1.0")
allowed_origins = [value.strip() for value in os.environ.get("TRAINER_ALLOWED_ORIGINS", "").split(",") if value.strip()]
if allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


class PlanRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2_000)


class TrainRequest(BaseModel):
    config: Dict[str, Any]


class PredictRequest(BaseModel):
    experiment: str
    records: list[dict] = Field(min_length=1, max_length=1_000)
    epoch: Optional[int] = Field(default=None, ge=1)


class GenerateRequest(BaseModel):
    experiment: str
    prompt: str = Field(min_length=1, max_length=2_000)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0, le=2)
    epoch: Optional[int] = Field(default=None, ge=1)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Auth is opt-in locally and enforced when TRAINER_API_KEY is configured."""
    if API_KEY and not (x_api_key and hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _json_safe(value):
    if is_dataclass(value): return _json_safe(asdict(value))
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try: return _json_safe(value.item())
        except Exception: pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)): return None
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    if hasattr(value, "__dict__"): return _json_safe(vars(value))
    return str(value)


def _safe_name(value: str, label: str = "experiment") -> str:
    if not NAME_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"{label} must be 1-64 letters, numbers, underscores, or hyphens.")
    return value


def _dataset_path(dataset_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", dataset_id): raise HTTPException(status_code=404, detail="Dataset not found.")
    folder = (UPLOAD_ROOT / dataset_id).resolve()
    if folder.parent != UPLOAD_ROOT or not folder.is_dir(): raise HTTPException(status_code=404, detail="Dataset not found.")
    files = [path for path in folder.iterdir() if path.is_file()]
    if len(files) != 1: raise HTTPException(status_code=404, detail="Dataset not found.")
    return files[0]


def _set_job(job_id: str, **updates):
    with JOBS_LOCK:
        if job_id in JOBS: JOBS[job_id].update(_json_safe(updates))


def _validate_training_config(raw: Dict[str, Any]):
    from trainer.core.config import TrainingConfig
    unknown = set(raw) - set(TrainingConfig.__dataclass_fields__)
    if unknown: raise HTTPException(status_code=400, detail=f"Unknown config fields: {', '.join(sorted(unknown))}")
    config = TrainingConfig.from_dict(raw)
    try: config.validate()
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = Path(config.dataset_path).resolve()
    if UPLOAD_ROOT not in path.parents or not path.is_file():
        raise HTTPException(status_code=400, detail="dataset_path must refer to a dataset uploaded through this API.")
    _safe_name(config.experiment_name)
    if config.epochs > MAX_EPOCHS or config.batch_size > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail="Requested epochs or batch size exceeds this server's training limit.")
    if any(not isinstance(size, int) or size < 1 or size > MAX_HIDDEN_UNITS for size in config.hidden_layers):
        raise HTTPException(status_code=400, detail="Each hidden layer must be an integer within the server limit.")
    return config


def _run_training(job_id: str, config):
    _set_job(job_id, status="running", message="Training started")
    try:
        from trainer.core import pipeline as pipeline_mod
        pipeline_mod.MODELS_ROOT, pipeline_mod.EXPERIMENTS_ROOT = str(MODEL_ROOT), str(EXPERIMENT_ROOT)
        result = pipeline_mod.run_pipeline(config, progress_cb=lambda metrics: _set_job(job_id, metrics=metrics, message=f"Epoch {metrics.get('epoch', '?')}"))
        _set_job(job_id, status="completed", result=result, message="Training completed")
    except Exception as exc:
        LOG.exception("training job %s failed", job_id)
        _set_job(job_id, status="failed", error="Training failed. Check server logs for details.", message=str(exc)[:300])


@app.get("/health")
def health(): return {"status": "ok", "trainer": "ready", "version": app.version}


@app.get("/api/system", dependencies=[Depends(require_api_key)])
def system():
    with JOBS_LOCK: active = sum(job["status"] in {"queued", "running"} for job in JOBS.values())
    return {"status": "ok", "active_jobs": active, "max_active_jobs": MAX_ACTIVE_JOBS}


@app.post("/api/datasets/upload", dependencies=[Depends(require_api_key)])
async def upload_dataset(file: UploadFile = File(...)):
    allowed, suffix = {".csv", ".tsv", ".json", ".jsonl", ".txt"}, Path(file.filename or "").suffix.lower()
    if suffix not in allowed: raise HTTPException(400, f"Unsupported file type. Use: {', '.join(sorted(allowed))}")
    dataset_id, folder = uuid.uuid4().hex, None
    try:
        folder = UPLOAD_ROOT / dataset_id; folder.mkdir(parents=True, exist_ok=False)
        destination, total = folder / f"dataset{suffix}", 0
        with destination.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES: raise HTTPException(413, "Dataset is too large.")
                out.write(chunk)
        return {"dataset_id": dataset_id, "filename": Path(file.filename).name, "bytes": total}
    except Exception:
        if folder:
            for child in folder.glob("*"): child.unlink(missing_ok=True)
            folder.rmdir()
        raise


@app.post("/api/datasets/analyze", dependencies=[Depends(require_api_key)])
def analyze_dataset(dataset_id: str):
    from trainer.datasets import analyzer, loader
    try:
        path = _dataset_path(dataset_id); fmt, data = loader.detect_format(str(path)), loader.load(str(path))
        stats = analyzer.analyze_text(data, path=str(path)) if fmt == "txt" else analyzer.analyze_tabular(data, path=str(path))
        return {"dataset_id": dataset_id, "format": fmt, "analysis": _json_safe(stats)}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/planner/plan", dependencies=[Depends(require_api_key)])
def create_plan(request: PlanRequest, dataset_id: str):
    from trainer.planner.nl_planner import plan
    try:
        result = plan(request.prompt, str(_dataset_path(dataset_id)))
        return {"plan": _json_safe(result["config"]), "reasoning": result.get("reasoning", []), "dataset_analysis": _json_safe(result.get("dataset_analysis", {}))}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/training/start", dependencies=[Depends(require_api_key)])
def start_training(request: TrainRequest):
    config = _validate_training_config(request.config)
    with JOBS_LOCK:
        if sum(job["status"] in {"queued", "running"} for job in JOBS.values()) >= MAX_ACTIVE_JOBS:
            raise HTTPException(429, detail="The trainer is busy. Wait for the current job to finish.")
        job_id = "job_" + uuid.uuid4().hex[:12]
        JOBS[job_id] = {"id": job_id, "status": "queued", "metrics": {}, "message": "Queued"}
    threading.Thread(target=_run_training, args=(job_id, config), daemon=True, name=job_id).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/training/{job_id}", dependencies=[Depends(require_api_key)])
def training_status(job_id: str):
    with JOBS_LOCK: job = JOBS.get(job_id)
    if job is None: raise HTTPException(404, "Training job not found.")
    return job


@app.get("/api/experiments", dependencies=[Depends(require_api_key)])
def list_experiments():
    from trainer.training.checkpoint import CheckpointManager
    out = []
    for folder in sorted(MODEL_ROOT.iterdir() if MODEL_ROOT.exists() else []):
        if folder.is_dir() and NAME_RE.fullmatch(folder.name):
            try:
                manager, latest = CheckpointManager(str(MODEL_ROOT), folder.name), None
                latest = manager.load_latest()
                if latest: out.append({"experiment": folder.name, "checkpoints": manager.list_checkpoints(), "algorithm": latest["metadata"].get("algorithm"), "metrics": latest["metadata"].get("metrics", {})})
            except Exception: LOG.warning("Skipping unreadable experiment %s", folder.name)
    return {"experiments": out}


@app.post("/api/predict", dependencies=[Depends(require_api_key)])
def predict(request: PredictRequest):
    from trainer.inference.predictor import Predictor
    try:
        name = _safe_name(request.experiment)
        return {"experiment": name, "predictions": _json_safe(Predictor(str(MODEL_ROOT), name, request.epoch).predict(request.records))}
    except HTTPException: raise
    except FileNotFoundError as exc: raise HTTPException(404, str(exc)) from exc
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/generate", dependencies=[Depends(require_api_key)])
def generate(request: GenerateRequest):
    from trainer.algorithms.transformer import TinyTransformer
    from trainer.tokenizers.char_tokenizer import CharTokenizer
    from trainer.training.checkpoint import CheckpointManager
    try:
        manager = CheckpointManager(str(MODEL_ROOT), _safe_name(request.experiment))
        checkpoint = manager.load_epoch(request.epoch) if request.epoch else manager.load_latest()
        if not checkpoint: raise FileNotFoundError("No trained checkpoint found.")
        model = TinyTransformer.from_state(checkpoint["model_state"]["transformer_state"])
        tokenizer = CharTokenizer.from_state(checkpoint["preprocessor_state"])
        return {"text": tokenizer.decode(model.generate(tokenizer.encode(request.prompt), request.max_new_tokens, request.temperature))}
    except HTTPException: raise
    except FileNotFoundError as exc: raise HTTPException(404, str(exc)) from exc
    except Exception as exc: raise HTTPException(400, str(exc)) from exc


WEB_ROOT = APP_ROOT / "web"
if WEB_ROOT.exists(): app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="web")
