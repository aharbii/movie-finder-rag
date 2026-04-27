# =============================================================================
# movie-finder-rag — Docker-only local development and runtime images
# =============================================================================

FROM python:3.13-slim AS uv-base

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy


FROM uv-base AS dev

ARG WITH_PROVIDERS=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONPATH="/workspace/src:/workspace"

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$WITH_PROVIDERS" ]; then \
        uv sync --all-groups --extra "$WITH_PROVIDERS" --active --no-install-project --no-install-workspace; \
    else \
        uv sync --all-groups --active --no-install-project --no-install-workspace; \
    fi

CMD ["sleep", "infinity"]


FROM uv-base AS builder

ARG WITH_PROVIDERS=""

WORKDIR /build

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$WITH_PROVIDERS" ]; then \
        uv sync --frozen --no-dev --extra "$WITH_PROVIDERS" --no-install-project --no-install-workspace; \
    else \
        uv sync --frozen --no-dev --no-install-project --no-install-workspace; \
    fi

COPY src/ src/
COPY scripts/ scripts/


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
