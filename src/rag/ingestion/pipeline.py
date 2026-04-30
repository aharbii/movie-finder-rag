import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rag.config import settings
from rag.embeddings.base import EmbeddingProvider
from rag.ingestion import csv_loader
from rag.models.movie import Movie
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore

_OUTPUT_ENV_PATH = Path("ingestion-outputs.env")
_REPORT_DIR = Path("outputs/reports")
_COST_REPORT_PATH = _REPORT_DIR / "cost-report.json"
_SKIPPED_MOVIES_REPORT_PATH = _REPORT_DIR / "skipped-movies.json"


def _embed_batch_with_fallback(
    movies: list[Movie],
    embedding_provider: EmbeddingProvider,
) -> tuple[list[Movie], list[list[float]], list[dict[str, Any]]]:
    """Embed a batch, falling back to per-movie embedding when batch calls fail."""
    logger = get_logger(__name__)
    vectors = embedding_provider.embed_batch([movie.plot for movie in movies])

    if len(vectors) == len(movies):
        return movies, vectors, []

    logger.warning(
        "Batch embedding returned %d vectors for %d movies. Falling back to per-movie embedding.",
        len(vectors),
        len(movies),
    )

    embedded_movies: list[Movie] = []
    embedded_vectors: list[list[float]] = []
    skipped_movies: list[dict[str, Any]] = []
    for movie in movies:
        vector = embedding_provider.embed(movie.plot)
        if not vector:
            skipped_movie = {
                "id": movie.id,
                "title": movie.title,
                "release_year": movie.release_year,
                "plot_chars": len(movie.plot),
                "reason": "embedding_failed_after_batch_fallback",
            }
            logger.error(
                "Skipping movie id=%s title=%r because embedding failed. plot_chars=%d",
                movie.id,
                movie.title,
                len(movie.plot),
            )
            skipped_movies.append(skipped_movie)
            continue
        embedded_movies.append(movie)
        embedded_vectors.append(vector)

    return embedded_movies, embedded_vectors, skipped_movies


def ingest_csv(embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
    """Orchestrate the offline RAG ingestion pipeline."""
    logger = get_logger(__name__)
    logger.info("Initializing RAG ingestion pipeline")
    movies = csv_loader.load_movies()
    if not movies:
        logger.warning("No movies found to ingest. Pipeline finished.")
        return

    batch_size = settings.batch_size
    model_info = embedding_provider.model_info
    target_name = vector_store.target_name(model_info)

    logger.info(
        "Loaded %d movies. provider=%s model=%s target=%s",
        len(movies),
        settings.embedding_provider,
        model_info.name,
        target_name,
    )

    inserted_movie_count = 0
    skipped_movies: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(movies), batch_size), desc="Ingesting Movies"):
        batch = movies[start : start + batch_size]
        embedded_movies, vectors, batch_skipped_movies = _embed_batch_with_fallback(
            batch, embedding_provider
        )
        skipped_movies.extend(batch_skipped_movies)
        if not vectors:
            logger.error(
                "Failed to generate embeddings for batch %d. Skipping.", start // batch_size
            )
            continue
        vector_store.upsert_batch(embedded_movies, vectors, model_info)
        inserted_movie_count += len(embedded_movies)

    usage = embedding_provider.get_model_usage()
    logger.info("INGESTION COMPLETE")
    logger.info("Total Movies Processed : %d", len(movies))
    logger.info("Total Movies Inserted  : %d", inserted_movie_count)
    logger.info("Total Movies Skipped   : %d", len(skipped_movies))
    logger.info("Total Prompt Tokens    : %d", usage.prompt_tokens)
    logger.info("Estimated USD Cost     : $%.5f", usage.estimated_cost_usd)

    _write_ingestion_outputs(target_name, model_info.name, model_info.dimension, usage)
    _write_skipped_movies_report(
        target_name=target_name,
        model_name=model_info.name,
        model_dimension=model_info.dimension,
        loaded_movie_count=len(movies),
        inserted_movie_count=inserted_movie_count,
        skipped_movies=skipped_movies,
    )


def _write_ingestion_outputs(
    target_name: str,
    model_name: str,
    dimension: int,
    usage: object,
) -> None:
    """Persist CI-friendly ingestion outputs and cost report artifacts."""
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0))
    total_tokens = int(getattr(usage, "total_tokens", 0))
    estimated_cost_usd = float(getattr(usage, "estimated_cost_usd", 0.0))

    _OUTPUT_ENV_PATH.write_text(
        "\n".join(
            [
                f"EMBEDDING_PROVIDER={settings.embedding_provider}",
                f"EMBEDDING_MODEL={model_name}",
                f"EMBEDDING_DIMENSION={dimension}",
                f"VECTOR_STORE={settings.vector_store}",
                f"VECTOR_STORE_TARGET_NAME={target_name}",
                f"VECTOR_COLLECTION_PREFIX={target_name}",
                f"INGESTION_PROMPT_TOKENS={prompt_tokens}",
                f"INGESTION_TOTAL_TOKENS={total_tokens}",
                f"INGESTION_ESTIMATED_COST_USD={estimated_cost_usd:.8f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _COST_REPORT_PATH.write_text(
        json.dumps(
            {
                "embedding_provider": settings.embedding_provider,
                "embedding_model": model_name,
                "embedding_dimension": dimension,
                "vector_store": settings.vector_store,
                "target_name": target_name,
                "collection_name": target_name,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_skipped_movies_report(
    target_name: str,
    model_name: str,
    model_dimension: int,
    loaded_movie_count: int,
    inserted_movie_count: int,
    skipped_movies: list[dict[str, Any]],
) -> None:
    """Persist a machine-readable report for movies skipped during ingestion."""
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _SKIPPED_MOVIES_REPORT_PATH.write_text(
        json.dumps(
            {
                "embedding_provider": settings.embedding_provider,
                "embedding_model": model_name,
                "embedding_dimension": model_dimension,
                "vector_store": settings.vector_store,
                "target_name": target_name,
                "loaded_movie_count": loaded_movie_count,
                "inserted_movie_count": inserted_movie_count,
                "skipped_movie_count": len(skipped_movies),
                "skipped_movies": skipped_movies,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
