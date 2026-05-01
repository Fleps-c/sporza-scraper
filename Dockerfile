# ── Stage 1: Build ────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the spaCy Dutch language model
RUN pip install --no-cache-dir --prefix=/install spacy && \
    PYTHONPATH=/install/lib/python3.12/site-packages \
    python -m spacy download nl_core_news_md

# ── Stage 2: Production ──────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /root /root

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home appuser && \
    mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

# Expose dashboard port
EXPOSE 8501

# Health check for dashboard service
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default: run the scraper CLI
ENTRYPOINT ["python", "-m", "sporza_scraper"]
