FROM python:3.11-slim

WORKDIR /app

COPY trainer.zip /tmp/trainer.zip
RUN apt-get update \
    && apt-get install -y --no-install-recommends unzip \
    && unzip -q /tmp/trainer.zip -d /app \
    && pip install --no-cache-dir -r /app/trainer/requirements-web.txt \
    && useradd --create-home --uid 10001 trainer \
    && mkdir -p /var/data \
    && chown -R trainer:trainer /app/trainer /var/data \
    && rm -rf /var/lib/apt/lists/* /tmp/trainer.zip

ENV TRAINER_DATA_DIR=/var/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
USER trainer
CMD ["sh", "-c", "uvicorn trainer.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
