# =============================================================================
# movie-finder-rag — Docker-only local development and runtime images
#
# Targets:
#   dev      Attached-container image used by docker-compose.yml and VS Code
#   builder  Intermediate dependency synchronization stage
#   runtime  One-shot ingestion image used by Jenkins and `make ingest`
# =============================================================================

FROM python:3.13-slim AS uv-base

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy


# ---- Stage 1: dev -----------------------------------------------------------
FROM uv-base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONPATH="/workspace/src"

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-groups --active --no-install-project --no-install-workspace

CMD ["sleep", "infinity"]


# ---- Stage 2: builder -------------------------------------------------------
FROM uv-base AS builder

WORKDIR /build

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-install-workspace

COPY src/ src/
COPY scripts/ scripts/


# ---- Stage 3: runtime -------------------------------------------------------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="movie-finder-rag"
LABEL org.opencontainers.image.source="https://github.com/aharbii/movie-finder-rag"
LABEL org.opencontainers.image.description="Movie Finder offline RAG ingestion pipeline"
LABEL org.opencontainers.image.licenses="MIT"

RUN useradd --system --uid 1001 --create-home --home-dir /home/appuser appuser

WORKDIR /app

COPY --link --from=builder /build/.venv /app/.venv
COPY --link --from=builder /build/src ./src
COPY --link --from=builder /build/scripts ./scripts

ENV HOME="/home/appuser" \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /app/dataset && chown -R appuser:appuser /app /home/appuser

USER appuser

ENTRYPOINT ["python", "-m", "rag.main"]
