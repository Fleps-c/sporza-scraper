# ── Stage 1: build dependencies ──────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install only production dependencies into a clean prefix
COPY requirements.txt pyproject.toml ./

# Create a separate requirements file without dev dependencies
RUN grep -v pytest requirements.txt > requirements-prod.txt && \
    pip install --no-cache-dir --prefix=/install -r requirements-prod.txt

# Pre-download the Dutch spaCy model into the install prefix
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m spacy download nl_core_news_md --target /install/lib/python3.11/site-packages


# ── Stage 2: production image ───────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy pre-built dependencies from builder
COPY --from=builder /install /usr/local

# Create a non-root user for security
RUN groupadd --gid 1000 scraper && \
    useradd  --uid 1000 --gid scraper --create-home scraper && \
    mkdir -p /app/data && chown -R scraper:scraper /app

# Copy application code (respects .dockerignore)
COPY --chown=scraper:scraper . .

# Switch to non-root user
USER scraper

# Expose Streamlit dashboard port
EXPOSE 8501

# Persist scraped data and SQLite DB across container restarts
VOLUME ["/app/data"]

# Health check for the Streamlit dashboard (only active when dashboard is running)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default: show CLI help. Override to run scraper or dashboard.
# Examples:
#   docker run sporza-scraper pl-scrape --limit 20
#   docker run -p 8501:8501 sporza-scraper streamlit run sporza_scraper/dashboard.py
CMD ["python", "-m", "sporza_scraper", "--help"]
