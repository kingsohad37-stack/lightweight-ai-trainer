from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import server
import local_dataset_model
from trainer.training.checkpoint import CheckpointManager

app = server.app

# server.py places the static root mount last. Keep API routes ahead of it.
_root_mount = getattr(server, "_root_mount", None)
if _root_mount is not None and _root_mount in app.router.routes:
    app.router.routes.remove(_root_mount)

# Replace the bundled trainer's dataset routes with the local open-source
# model implementation. This requires no provider API key.
for route in list(app.router.routes):
    if getattr(route, "path", None) in {"/api/ai/dataset", "/api/ai/dataset/generate"}:
        app.router.routes.remove(route)

app.router.add_api_route(
    "/api/ai/dataset/generate",
    local_dataset_model.generate_dataset,
    methods=["POST"],
    dependencies=[Depends(server.original.require_api_key)],
)
app.router.add_api_route(
    "/api/ai/dataset",
    local_dataset_model.generate_dataset,
    methods=["POST"],
    dependencies=[Depends(server.original.require_api_key)],
)

_dataset_routes = [
    route for route in app.router.routes
    if getattr(route, "path", None) in {"/api/ai/dataset", "/api/ai/dataset/generate"}
    and "POST" in getattr(route, "methods", set())
]
for route in _dataset_routes:
    app.router.routes.remove(route)
for route in reversed(_dataset_routes):
    app.router.routes.insert(0, route)


def _safe_export_path(base: Path, child: Path) -> Path:
    resolved = child.resolve()
    if base.resolve() not in resolved.parents:
        raise HTTPException(400, "Invalid experiment path.")
    return resolved


# ---------------- Automatic two-step workflow ----------------

class AutoPlanRequest(BaseModel):
    description: str = Field(min_length=5, max_length=20_000)


class AutoBuildRequest(BaseModel):
    description: str = Field(min_length=5, max_length=20_000)
    plan: dict = Field(default_factory=dict)


_auto_jobs: dict[str, dict] = {}
_auto_lock = threading.Lock()


def _heuristic_plan(description: str) -> dict:
    text = description.lower()
    chatbot = any(x in text for x in ("chatbot", "chat bot", "conversational", "assistant", "conversation", "customer support"))
    classification = any(x in text for x in ("classify", "classification", "category", "categorize", "predict a label", "sentiment"))
    regression = any(x in text for x in ("regression", "predict a price", "predict a number", "numeric prediction"))
    if chatbot:
        return {
            "task": "language_modeling",
            "algorithm": "tiny_transformer",
            "dataset_columns": ["system", "user", "assistant"],
            "dataset_format": "txt",
            "rows": 100,
            "reason": "The description asks for a conversational assistant, so a tiny-transformer language model is the best automatic path.",
        }
    if regression:
        return {
            "task": "regression",
            "algorithm": "auto",
            "dataset_columns": ["text", "target"],
            "dataset_format": "csv",
            "rows": 100,
            "reason": "The description asks for a numeric prediction task.",
        }
    if classification:
        return {
            "task": "classification",
            "algorithm": "auto",
            "dataset_columns": ["text", "label"],
            "dataset_format": "csv",
            "rows": 100,
            "reason": "The description asks for categories or labels.",
        }
    return {
        "task": "language_modeling",
        "algorithm": "tiny_transformer",
        "dataset_columns": ["system", "user", "assistant"],
        "dataset_format": "txt",
        "rows": 100,
        "reason": "No structured target was specified, so the automatic mode treats the request as a conversational/text-generation model.",
    }


def _set_auto(job_id: str, **values):
    with _auto_lock:
        job = _auto_jobs.setdefault(job_id, {"job_id": job_id})
        job.update(values)


async def _internal_post(path: str, *, headers: dict | None = None, json_body: dict | None = None, params: dict | None = None, files=None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://trainitlocal") as client:
        response = await client.post(path, headers=headers or {}, json=json_body, params=params, files=files)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(str(detail))
        return response.json()


def _run_auto_build(job_id: str, description: str, plan: dict, auth_headers: dict):
    try:
        _set_auto(job_id, status="running", message="Generating your training data locally…", stage="dataset")
        selected = plan or _heuristic_plan(description)
        columns = selected.get("dataset_columns") or ["system", "user", "assistant"]
        fmt = selected.get("dataset_format") or "txt"
        rows = int(selected.get("rows") or 100)
        rows = max(25 if selected.get("task") in {"classification", "regression"} else 5, min(rows, 500))
        dataset = asyncio.run(_internal_post(
            "/api/ai/dataset/generate",
            headers=auth_headers,
            json_body={"topic": description, "columns": ",".join(columns), "rows": rows, "format": fmt},
        ))

        _set_auto(job_id, message="Loading the generated dataset…", stage="upload")
        upload = asyncio.run(_internal_post(
            "/api/datasets/upload",
            headers=auth_headers,
            files={"file": (dataset["filename"], dataset["content"].encode("utf-8"), dataset.get("media_type", "text/plain"))},
        ))
        dataset_id = upload["dataset_id"]
        asyncio.run(_internal_post(
            "/api/datasets/analyze",
            headers=auth_headers,
            params={"dataset_id": dataset_id},
        ))

        _set_auto(job_id, message="Creating the training plan…", stage="planning", dataset_id=dataset_id)
        planned = asyncio.run(_internal_post(
            "/api/planner/plan",
            headers=auth_headers,
            params={"dataset_id": dataset_id},
            json_body={"prompt": description},
        ))
        config = planned.get("plan") or selected
        if not isinstance(config, dict):
            raise RuntimeError("The planner returned an invalid training configuration.")
        config = dict(config)
        experiment = server.original._safe_name(
            config.get("experiment_name") or "auto-" + uuid.uuid4().hex[:10]
        )
        config["experiment_name"] = experiment

        _set_auto(job_id, message="Training your AI model…", stage="training", config=config, experiment=experiment)
        started = asyncio.run(_internal_post(
            "/api/training/start",
            headers=auth_headers,
            json_body={"config": config},
        ))
        training_job_id = started.get("job_id")
        if not training_job_id:
            raise RuntimeError("Training service did not return a job ID.")

        while True:
            status = asyncio.run(_internal_post(
                "/api/training/" + str(training_job_id), headers=auth_headers
            ))
            state = status.get("status")
            _set_auto(job_id, message=f"Training: {state}…", stage="training", training_job_id=training_job_id)
            if state == "completed":
                break
            if state in {"failed", "error", "cancelled"}:
                raise RuntimeError(status.get("error") or status.get("message") or "Training failed.")
            import time
            time.sleep(1.5)

        _set_auto(job_id, message="Packaging your deployable AI…", stage="export")
        _set_auto(job_id, status="completed", message="Your deployable AI is ready.", stage="done", result={
            "experiment": experiment,
            "dataset_id": dataset_id,
            "training_job_id": training_job_id,
            "download_url": f"/api/experiments/{experiment}/download",
            "config": config,
            "plan": selected,
        })
    except Exception as exc:
        _set_auto(job_id, status="failed", message="Automatic build failed.", stage="error", error=str(exc))


@app.post("/api/auto/plan", dependencies=[Depends(server.original.require_api_key)])
def auto_plan(request: AutoPlanRequest):
    return {"plan": _heuristic_plan(request.description)}


@app.post("/api/auto/build", dependencies=[Depends(server.original.require_api_key)])
def auto_build(request: AutoBuildRequest, http_request: Request):
    job_id = uuid.uuid4().hex
    auth = {}
    if http_request.headers.get("x-api-key"):
        auth["X-API-Key"] = http_request.headers["x-api-key"]
    _set_auto(job_id, status="queued", message="Starting automatic build…", stage="queued")
    thread = threading.Thread(
        target=_run_auto_build,
        args=(job_id, request.description, request.plan, auth),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/auto/status/{job_id}", dependencies=[Depends(server.original.require_api_key)])
def auto_status(job_id: str):
    with _auto_lock:
        job = _auto_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Automatic build job not found.")
    return dict(job)


# Replace server.py's checkpoint-only download with a self-contained export.
for route in list(app.routes):
    if getattr(route, "path", None) == "/api/experiments/{experiment}/download" and "GET" in getattr(route, "methods", set()):
        app.router.routes.remove(route)


@server.original.app.get("/api/experiments/{experiment}/download", dependencies=[Depends(server.original.require_api_key)])
def download_experiment(experiment: str):
    name = server.original._safe_name(experiment)
    manager = CheckpointManager(str(server.original.MODEL_ROOT), name)
    latest = manager.load_latest()
    if not latest:
        raise HTTPException(404, "No trained model found for this experiment.")

    model_dir = _safe_export_path(server.original.MODEL_ROOT, server.original.MODEL_ROOT / name)
    runtime_dir = Path(__import__("trainer").__file__).resolve().parent
    root = Path(__file__).resolve().parent
    launcher = root / "run_chat.py"
    runtime_requirements = root / "requirements-runtime.txt"

    temp = tempfile.NamedTemporaryFile(prefix=f"{name}-export-", suffix=".zip", dir=str(server.original.MODEL_ROOT), delete=False)
    temp_path = Path(temp.name)
    temp.close()
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in model_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, Path("model") / path.relative_to(model_dir))
            if launcher.is_file():
                archive.write(launcher, "run_chat.py")
            if runtime_requirements.is_file():
                archive.write(runtime_requirements, "requirements-runtime.txt")
            for path in runtime_dir.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, Path("runtime/trainer") / path.relative_to(runtime_dir))
            manifest = {
                "experiment": name,
                "algorithm": latest["metadata"].get("algorithm"),
                "epoch": latest["metadata"].get("epoch"),
                "metrics": latest["metadata"].get("metrics", {}),
                "config": latest["metadata"].get("config", {}),
                "standalone_runtime": latest["metadata"].get("algorithm") == "tiny_transformer",
                "run_command": "python run_chat.py" if latest["metadata"].get("algorithm") == "tiny_transformer" else None,
            }
            archive.writestr("MODEL_MANIFEST.json", json.dumps(manifest, indent=2))
            archive.writestr(
                "README.txt",
                "Extract this ZIP. For a tiny-transformer language model, install requirements-runtime.txt and run python run_chat.py.\n"
                "The model checkpoint is in model/ and its bundled runtime is in runtime/trainer/.\n",
            )
        def stream():
            try:
                with temp_path.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                temp_path.unlink(missing_ok=True)
        headers = {"Content-Disposition": f'attachment; filename="{name}-trained-model.zip"'}
        return StreamingResponse(stream(), media_type="application/zip", headers=headers)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


if _root_mount is not None:
    app.router.routes.append(_root_mount)
