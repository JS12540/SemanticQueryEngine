# ─────────────────────────────────────────────────────────────────────────────
# Cashflo NLP-to-SQL  –  Dockerfile
#
# Builds a production image that runs the Streamlit application.
# Everything is accessible through the browser UI – no CLI.
#
# Build:
#   docker build -t cashflo-nlp .
#
# Run:
#   docker run -p 8501:8501 --env-file .env cashflo-nlp
#   # then open http://localhost:8501
#
# Pass only the API key:
#   docker run -p 8501:8501 -e OPENAI_API_KEY=sk-proj-... cashflo-nlp
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /usr/local/bin/uv

WORKDIR /app

# Layer-cache: copy manifests before source code
COPY pyproject.toml ./

# Create venv and install all runtime deps via uv (no dev extras)
RUN uv venv .venv && \
    uv sync --no-dev --no-install-project

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN addgroup --system cashflo && \
    adduser  --system --ingroup cashflo cashflo

WORKDIR /app

# Venv from builder
COPY --from=builder /app/.venv /app/.venv

# Application source (no main.py – Streamlit only)
COPY engine/             ./engine/
COPY app.py              ./app.py
COPY semantic_layer.yaml ./semantic_layer.yaml
COPY cashflo_sample.db   ./cashflo_sample.db

# Streamlit server config
COPY .streamlit/ ./.streamlit/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1   \
    PYTHONUNBUFFERED=1

RUN chown -R cashflo:cashflo /app
USER cashflo

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; \
         urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501",         \
     "--server.address=0.0.0.0",   \
     "--server.headless=true"]
