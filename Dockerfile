# =============================================================================
# rag_ingestion — batch job Dockerfile
#
# This is a one-shot batch container, not a long-running server.
# Run via docker compose run ingestion  or  docker run <image>
#
# Build context: rag_ingestion/  (standalone — has its own uv.lock)
# =============================================================================

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy dependency manifests first — optimal layer caching
COPY pyproject.toml uv.lock ./

# Sync only production dependencies into /build/.venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="rag-ingestion"
LABEL org.opencontainers.image.description="Movie dataset ingestion pipeline — embeds and loads into Qdrant"

# Non-root user for security
RUN useradd --system --uid 1001 --no-create-home appuser

WORKDIR /app

# Copy the virtualenv and source code
COPY --from=builder /build/.venv /app/.venv
COPY src/ src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# Batch job entry point — runs the full ingestion pipeline
ENTRYPOINT ["python", "-m", "src.main"]
