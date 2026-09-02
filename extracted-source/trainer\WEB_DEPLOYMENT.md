# Lightweight AI Trainer — Web Deployment

This package wraps the existing `trainer/` ML engine with a FastAPI web API and a browser UI.

## 1. Run locally

From the directory containing the `trainer/` folder:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r trainer/requirements-web.txt
```

Start the server:

```bash
uvicorn trainer.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## 2. Docker

From the directory containing `trainer/`:

```bash
docker build -t lightweight-ai-trainer trainer
docker run --rm -p 8000:8000 lightweight-ai-trainer
```

Open `http://127.0.0.1:8000`.

## 3. Render

The `render.yaml` and `Dockerfile` are included. The Docker image starts the
package as `trainer.api.main`, matching the imports used by the app.

Connect the repository to Render and deploy the Docker web service. The free
plan is suitable only for demos: Render's filesystem is ephemeral, so uploads,
checkpoints, and job history are lost after a restart or deploy. For production,
use object storage plus a durable job queue, or attach a paid persistent disk
at `/var/data` (one instance only).

## 4. Web flow

1. Upload CSV/TSV/JSON/JSONL/TXT.
2. The API saves the dataset under the deployment data directory.
3. Dataset analysis runs using the existing loader/analyzer.
4. The planner receives the uploaded dataset plus the user's natural-language request.
5. The planner returns a real `TrainingConfig`, reasoning, and dataset analysis.
6. Training starts as a background job.
7. The existing `run_pipeline()` performs the actual training.
8. Job status and metrics are available through `/api/training/{job_id}`.

## Important production notes

The API applies upload/training limits, verifies dataset paths stay within its
uploads folder, validates experiment names, and supports an optional
`TRAINER_API_KEY` sent as `X-API-Key`. Set that key before public deployment.
It also limits the demo server to one active job. For multi-user production,
replace the in-process worker with a Redis/Celery/RQ worker, persist job state,
and use object storage. Do not treat an ephemeral filesystem as permanent model storage.
