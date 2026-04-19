import argparse
import json
import logging
import os
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import chromadb
import numpy as np
from qdrant_client import QdrantClient

BACKUP_ROOT = Path("outputs/backups/chromadb")
type BackupMetadataValue = str | int | float | bool | None | list[str | int | float | bool]


@dataclass(slots=True)
class BackupRecord:
    """Portable backup record representation used by script-local adapters."""

    id: str
    vector: list[float]
    payload_json: str


@dataclass(slots=True)
class BackupConfig:
    """Script-local backup configuration resolved from CLI args and env."""

    vector_store: str
    collection_name: str
    output_root: Path
    batch_size: int
    vector_store_url: str | None = None
    vector_store_api_key: str | None = None


@dataclass(slots=True)
class BackupStats:
    """Summary of a completed vector store backup."""

    vector_store: str
    collection_name: str
    output_path: Path
    remote_points: int
    local_points: int
    artifact_size_bytes: int


class VectorStoreBackupSource(Protocol):
    """Contract for script-local vector store backup adapters."""

    def count_points(self) -> int:
        """Return the total number of points in the source collection."""

    def iter_batches(self) -> Iterator[list[BackupRecord]]:
        """Yield source records in batches ready to persist into the backup artifact."""


class QdrantBackupSource:
    """Qdrant backup adapter used by the current script."""

    def __init__(self, config: BackupConfig) -> None:
        if not config.vector_store_url:
            raise ValueError("Qdrant backup requires VECTOR_STORE_URL or QDRANT_URL.")
        if not config.vector_store_api_key:
            raise ValueError(
                "Qdrant backup requires VECTOR_STORE_API_KEY or QDRANT_API_KEY_RW."
            )

        self.config = config
        self.client = QdrantClient(
            url=config.vector_store_url,
            api_key=config.vector_store_api_key,
        )

    def count_points(self) -> int:
        """Return the total number of Qdrant points in the configured collection."""
        return int(
            self.client.count(collection_name=self.config.collection_name, exact=True).count
        )

    def iter_batches(self) -> Iterator[list[BackupRecord]]:
        """Yield Qdrant points as generic backup records."""
        offset: Any = None

        while True:
            records, next_page = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=self.config.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not records:
                return

            batch: list[BackupRecord] = []
            for record in records:
                batch.append(
                    BackupRecord(
                        id=str(record.id),
                        vector=_coerce_vector(record.vector, record.id),
                        payload_json=_serialize_payload(record.payload, record.id),
                    )
                )

            yield batch

            if next_page is None:
                return
            offset = next_page


def backup() -> None:
    """CLI entrypoint for vector store backup."""
    configure_logging()
    backup_vector_store(load_config())


def backup_vector_store(config: BackupConfig) -> BackupStats:
    """Back up the configured vector store collection into a local ChromaDB artifact."""
    logger = logging.getLogger("backup_script")
    source = build_backup_source(config)
    output_path = _backup_path(config)

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(output_path))
    collection = chroma_client.get_or_create_collection(name=config.collection_name)

    remote_points = source.count_points()
    logger.info(
        "Starting %s backup for collection '%s' into '%s'.",
        config.vector_store,
        config.collection_name,
        output_path,
    )

    remote_ids: set[str] = set()
    for batch in source.iter_batches():
        ids = [record.id for record in batch]
        remote_ids.update(ids)
        embeddings = [record.vector for record in batch]
        metadatas: list[dict[str, BackupMetadataValue]] = [
            {"payload_json": record.payload_json} for record in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            metadatas=cast(Any, metadatas),
        )

    local_points = int(collection.count())
    local_ids = {str(record_id) for record_id in collection.get()["ids"]}

    if local_points != remote_points:
        raise RuntimeError(
            "Backup validation failed: "
            f"remote collection has {remote_points} points but local ChromaDB has {local_points}."
        )

    if local_ids != remote_ids:
        raise RuntimeError(
            "Backup validation failed: local ChromaDB IDs do not match the remote vector store IDs."
        )

    artifact_size_bytes = _directory_size(output_path)
    logger.info(
        "Backup completed for '%s/%s': %d points saved to %s (%s, %d bytes).",
        config.vector_store,
        config.collection_name,
        local_points,
        output_path,
        _format_size(artifact_size_bytes),
        artifact_size_bytes,
    )

    return BackupStats(
        vector_store=config.vector_store,
        collection_name=config.collection_name,
        output_path=output_path,
        remote_points=remote_points,
        local_points=local_points,
        artifact_size_bytes=artifact_size_bytes,
    )


def build_backup_source(config: BackupConfig) -> VectorStoreBackupSource:
    """Resolve the script-local vector store adapter from the provided config."""
    if config.vector_store == "qdrant":
        return QdrantBackupSource(config)

    raise ValueError(f"Unsupported vector store for backup: {config.vector_store}")


def load_config(argv: list[str] | None = None) -> BackupConfig:
    """Resolve script configuration from CLI args with CI-friendly env fallbacks."""
    vector_store = _env_or_default("VECTOR_STORE", "qdrant")
    collection_name = (
        _env_or_none("BACKUP_COLLECTION_NAME")
        or _env_or_none("QDRANT_COLLECTION_NAME")
        or "movies"
    )
    output_root = _env_or_default("BACKUP_OUTPUT_ROOT", str(BACKUP_ROOT))
    batch_size = int(_env_or_default("BACKUP_BATCH_SIZE", "1000"))
    vector_store_url = _env_or_none("VECTOR_STORE_URL") or _env_or_none("QDRANT_URL")
    vector_store_api_key = _env_or_none("VECTOR_STORE_API_KEY") or _env_or_none(
        "QDRANT_API_KEY_RW"
    )

    parser = argparse.ArgumentParser(
        description="Back up a configured vector store collection into a local ChromaDB artifact."
    )
    parser.add_argument(
        "--vector-store",
        default=vector_store,
        help="Vector store type to back up. Defaults to VECTOR_STORE or 'qdrant'.",
    )
    parser.add_argument(
        "--collection-name",
        default=collection_name,
        help="Collection name to back up.",
    )
    parser.add_argument(
        "--output-root",
        default=output_root,
        help="Root directory for backup artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help="Number of records to fetch per source batch.",
    )
    parser.add_argument(
        "--vector-store-url",
        default=vector_store_url,
        help="Vector store endpoint URL.",
    )
    parser.add_argument(
        "--vector-store-api-key",
        default=vector_store_api_key,
        help="Vector store API key.",
    )

    args = parser.parse_args(argv)
    return BackupConfig(
        vector_store=args.vector_store,
        collection_name=args.collection_name,
        output_root=Path(args.output_root),
        batch_size=args.batch_size,
        vector_store_url=args.vector_store_url,
        vector_store_api_key=args.vector_store_api_key,
    )


def configure_logging() -> None:
    """Configure stdlib logging for script execution."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _env_or_none(name: str) -> str | None:
    """Return an env var only when it is a non-empty string."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_or_default(name: str, default: str) -> str:
    """Return a non-empty env var value or the provided default."""
    return _env_or_none(name) or default


def _backup_path(config: BackupConfig) -> Path:
    """Return the backup artifact directory for a vector store collection."""
    return config.output_root / config.vector_store / config.collection_name.replace("/", "-")


def _serialize_payload(payload: Any, record_id: Any) -> str:
    """Serialize a vector store payload into JSON for Chroma metadata storage."""
    if not isinstance(payload, dict):
        raise ValueError(f"Source record '{record_id}' is missing a payload.")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _coerce_vector(vector: Any, record_id: Any) -> list[float]:
    """Normalize vector store payloads into a single dense vector."""
    if vector is None:
        raise ValueError(f"Source record '{record_id}' is missing a vector.")

    if isinstance(vector, Mapping):
        if len(vector) != 1:
            raise ValueError(
                f"Source record '{record_id}' uses multiple named vectors, which backup does not support."
            )
        vector = next(iter(vector.values()))

    if not isinstance(vector, list):
        raise ValueError(f"Source record '{record_id}' returned an unsupported vector format.")

    return [float(value) for value in vector]


def _directory_size(path: Path) -> int:
    """Return the total size in bytes for all files within a directory tree."""
    if not path.exists():
        return 0
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _format_size(size_bytes: int) -> str:
    """Render a byte size in a compact human-readable format."""
    if size_bytes == 0:
        return "0 B"

    size = float(size_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size_bytes} B"


if __name__ == "__main__":
    backup()
