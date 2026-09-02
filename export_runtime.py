from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse

import server
from trainer.training.checkpoint import CheckpointManager

app = server.app

# server.py places the static root mount last. Keep API routes ahead of it.
_root_mount = getattr(server, "_root_mount", None)
if _root_mount is not None and _root_mount in app.router.routes:
    app.router.routes.remove(_root_mount)

# Re-register dataset generation explicitly at runtime. This removes any
# duplicate/stale route definitions inherited from the bundled trainer and
# guarantees both supported POST endpoints resolve to the same handler.
for route in list(app.router.routes):
    if getattr(route, "path", None) in {"/api/ai/dataset", "/api/ai/dataset/generate"}:
        app.router.routes.remove(route)

app.router.add_api_route(
    "/api/ai/dataset",
    server.generate_dataset,
    methods=["POST"],
    dependencies=[Depends(server.original.require_api_key)],
)
app.router.add_api_route(
    "/api/ai/dataset/generate",
    server.generate_dataset,
    methods=["POST"],
    dependencies=[Depends(server.original.require_api_key)],
)

# Replace server.py's checkpoint-only download with a self-contained export.
for route in list(app.routes):
    if getattr(route, "path", None) == "/api/experiments/{experiment}/download" and "GET" in getattr(route, "methods", set()):
        app.router.routes.remove(route)


def _safe_export_path(base: Path, child: Path) -> Path:
    resolved = child.resolve()
    if base.resolve() not in resolved.parents:
        raise HTTPException(400, "Invalid experiment path.")
    return resolved


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

    # Build on a temporary file instead of holding a potentially large model
    # export entirely in RAM.
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
