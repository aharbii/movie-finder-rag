import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from rag.chunking.base import Chunker, ChunkResult
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
_CHUNK_ID_MULTIPLIER = 1_000_000


def _embed_batch_with_fallback(
    chunks: list[Movie],
    embedding_provider: EmbeddingProvider,
) -> tuple[list[Movie], list[list[float]], list[dict[str, Any]]]:
    """Embed a batch, falling back to per-chunk embedding when batch calls fail."""
    logger = get_logger(__name__)
    vectors = embedding_provider.embed_batch([chunk.plot for chunk in chunks])

    if len(vectors) == len(chunks):
        return chunks, vectors, []

    logger.warning(
        "Batch embedding returned %d vectors for %d chunks. Falling back to per-chunk embedding.",
        len(vectors),
        len(chunks),
    )

    embedded_chunks: list[Movie] = []
    embedded_vectors: list[list[float]] = []
    skipped_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        vector = embedding_provider.embed(chunk.plot)
        if not vector:
            skipped_chunk = {
                "id": chunk.id,
                "source_movie_id": chunk.source_movie_id,
                "title": chunk.title,
                "release_year": chunk.release_year,
                "chunk_index": chunk.chunk_index,
                "chunk_strategy": chunk.chunk_strategy,
                "plot_chars": len(chunk.plot),
                "reason": "embedding_failed_after_batch_fallback",
            }
            logger.error(
                "Skipping chunk id=%s source_movie_id=%s title=%r because embedding failed. "
                "plot_chars=%d",
                chunk.id,
                chunk.source_movie_id,
                chunk.title,
                len(chunk.plot),
            )
            skipped_chunks.append(skipped_chunk)
            continue
        embedded_chunks.append(chunk)
        embedded_vectors.append(vector)

    return embedded_chunks, embedded_vectors, skipped_chunks


def ingest_csv(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chunker: Chunker,
) -> None:
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
    chunks = _chunk_movies(movies, chunker)
    if not chunks:
        logger.warning("No chunks generated to ingest. Pipeline finished.")
        return

    logger.info(
        "Loaded %d movies and generated %d chunks. provider=%s model=%s target=%s chunker=%s",
        len(movies),
        len(chunks),
        settings.embedding_provider,
        model_info.name,
        target_name,
        chunker.strategy_name,
    )

    inserted_chunk_count = 0
    skipped_chunks: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(chunks), batch_size), desc="Ingesting Movie Chunks"):
        batch = chunks[start : start + batch_size]
        embedded_chunks, vectors, batch_skipped_chunks = _embed_batch_with_fallback(
            batch, embedding_provider
        )
        skipped_chunks.extend(batch_skipped_chunks)
        if not vectors:
            logger.error(
                "Failed to generate embeddings for batch %d. Skipping.", start // batch_size
            )
            continue
        vector_store.upsert_batch(embedded_chunks, vectors, model_info)
        inserted_chunk_count += len(embedded_chunks)

    usage = embedding_provider.get_model_usage()
    logger.info("INGESTION COMPLETE")
    logger.info("Total Movies Processed : %d", len(movies))
    logger.info("Total Chunks Generated : %d", len(chunks))
    logger.info("Total Chunks Inserted  : %d", inserted_chunk_count)
    logger.info("Total Chunks Skipped   : %d", len(skipped_chunks))
    logger.info("Total Prompt Tokens    : %d", usage.prompt_tokens)
    logger.info("Estimated USD Cost     : $%.5f", usage.estimated_cost_usd)

    _write_ingestion_outputs(
        target_name,
        model_info.name,
        model_info.dimension,
        usage,
        chunker.strategy_name,
    )
    _write_skipped_movies_report(
        target_name=target_name,
        model_name=model_info.name,
        model_dimension=model_info.dimension,
        loaded_movie_count=len(movies),
        generated_chunk_count=len(chunks),
        inserted_chunk_count=inserted_chunk_count,
        chunking_strategy=chunker.strategy_name,
        skipped_chunks=skipped_chunks,
    )


def _chunk_movies(movies: list[Movie], chunker: Chunker) -> list[Movie]:
    chunks: list[Movie] = []
    for movie in movies:
        movie_chunks = chunker.chunk(movie)
        chunk_count = len(movie_chunks)
        for chunk in movie_chunks:
            chunks.append(_movie_from_chunk(movie, chunk, chunk_count, chunker.strategy_name))
    return chunks


def _movie_from_chunk(
    movie: Movie,
    chunk: ChunkResult,
    chunk_count: int,
    chunk_strategy: str,
) -> Movie:
    chunk_id = movie.id if chunk_count == 1 else movie.id * _CHUNK_ID_MULTIPLIER + chunk.chunk_index
    chunk_field = chunk.payload.get("chunk_field")
    return movie.model_copy(
        update={
            "id": chunk_id,
            "plot": chunk.text,
            "source_movie_id": movie.id,
            "chunk_index": chunk.chunk_index,
            "chunk_count": chunk_count,
            "chunk_strategy": chunk_strategy,
            "chunk_field": str(chunk_field) if chunk_field is not None else None,
        }
    )


def _write_ingestion_outputs(
    target_name: str,
    model_name: str,
    dimension: int,
    usage: object,
    chunking_strategy: str,
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
                f"CHUNKING_STRATEGY={chunking_strategy}",
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
                "chunking_strategy": chunking_strategy,
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
    generated_chunk_count: int,
    inserted_chunk_count: int,
    chunking_strategy: str,
    skipped_chunks: list[dict[str, Any]],
) -> None:
    """Persist a machine-readable report for chunks skipped during ingestion."""
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
                "generated_chunk_count": generated_chunk_count,
                "inserted_chunk_count": inserted_chunk_count,
                "chunking_strategy": chunking_strategy,
                "skipped_chunk_count": len(skipped_chunks),
                "skipped_chunks": skipped_chunks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
