# CrediTrust Complaint RAG — single image, two entrypoints (API or Streamlit UI)
# selected via docker-compose `command:` (see docker-compose.yml).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by chromadb/torch wheels at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py ./app.py

# Non-root runtime user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/vector_store /app/logs \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["uvicorn", "creditrust.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
