FROM python:3.11-slim

WORKDIR /app

COPY trainer.zip /tmp/trainer.zip
COPY extract_trainer.py /tmp/extract_trainer.py
COPY server.py /app/server.py
COPY local_dataset_model.py /app/local_dataset_model.py
COPY export_runtime.py /app/export_runtime.py
COPY run_chat.py /app/run_chat.py
COPY requirements-runtime.txt /app/requirements-runtime.txt
COPY web-overrides /tmp/web-overrides
RUN python /tmp/extract_trainer.py
RUN cp /tmp/web-overrides/index.html /app/trainer/web/index.html \
    && cp /tmp/web-overrides/app.js /app/trainer/web/app.js \
    && cp /tmp/web-overrides/styles.css /app/trainer/web/styles.css
RUN pip install --no-cache-dir -r /app/trainer/requirements-web.txt \
    && pip install --no-cache-dir "transformers>=4.56,<5" "safetensors>=0.5,<1"

# Bundle a small instruction-tuned open-source model locally so dataset
# generation works without Gemini/OpenAI/Groq credentials at runtime.
ENV HF_HOME=/opt/huggingface
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='HuggingFaceTB/SmolLM2-135M-Instruct', local_dir='/opt/local-model')"

RUN useradd --create-home --uid 10001 trainer \
    && mkdir -p /var/data \
    && chown -R trainer:trainer /app/trainer /app/server.py /app/local_dataset_model.py /app/export_runtime.py /app/run_chat.py /app/requirements-runtime.txt /opt/local-model /var/data \
    && rm -rf /tmp/trainer.zip /tmp/extract_trainer.py /tmp/web-overrides

ENV TRAINER_DATA_DIR=/var/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
USER trainer
CMD ["sh", "-c", "uvicorn export_runtime:app --host 0.0.0.0 --port ${PORT:-8000}"]
