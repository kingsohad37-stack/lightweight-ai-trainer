FROM python:3.11-slim

WORKDIR /app

COPY trainer.zip /tmp/trainer.zip
COPY extract_trainer.py /tmp/extract_trainer.py
COPY server.py /app/server.py
COPY web-overrides /tmp/web-overrides
RUN python /tmp/extract_trainer.py
RUN cp /tmp/web-overrides/index.html /app/trainer/web/index.html \
    && cp /tmp/web-overrides/app.js /app/trainer/web/app.js \
    && cp /tmp/web-overrides/styles.css /app/trainer/web/styles.css
RUN pip install --no-cache-dir -r /app/trainer/requirements-web.txt
RUN useradd --create-home --uid 10001 trainer \
    && mkdir -p /var/data \
    && chown -R trainer:trainer /app/trainer /app/server.py /var/data \
    && rm -rf /tmp/trainer.zip /tmp/extract_trainer.py /tmp/web-overrides

ENV TRAINER_DATA_DIR=/var/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
USER trainer
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
