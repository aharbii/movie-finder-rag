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

from rag.vectorstore.naming import sanitize_collection_token

BACKUP_ROOT = Path("outputs/backups")
INGESTION_OUTPUTS_PATH = Path("ingestion-outputs.env")
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
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    vector_store_url: str | None = None
    vector_store_api_key: str | None = None
    chromadb_persist_path: str | None = None
    pinecone_index_name: str | None = None
    pinecone_index_host: str | None = None
    pgvector_dsn: str | None = None
    pgvector_schema: str = "public"


@dataclass(slots=True)
class BackupStats:
    """Summary of a completed vector store backup."""

    vector_store: str
    collection_name: str
    output_path: Path
    remote_points: int
    local_points: int
    artifact_size_bytes: int
    manifest_path: Path


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
            raise ValueError("Qdrant backup requires VECTOR_STORE_API_KEY or QDRANT_API_KEY_RW.")

        self.config = config
        self.client = QdrantClient(
            url=config.vector_store_url,
            api_key=config.vector_store_api_key,
        )

    def count_points(self) -> int:
        """Return the total number of Qdrant points in the configured collection."""
        return int(self.client.count(collection_name=self.config.collection_name, exact=True).count)

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


class ChromaDBBackupSource:
    """ChromaDB backup adapter used for local persistent collections."""

    def __init__(self, config: BackupConfig) -> None:
        if not config.chromadb_persist_path:
            raise ValueError("ChromaDB backup requires CHROMADB_PERSIST_PATH.")

        self.config = config
        self.client = chromadb.PersistentClient(path=config.chromadb_persist_path)
        self.collection = self.client.get_collection(name=config.collection_name)

    def count_points(self) -> int:
        """Return the total number of ChromaDB vectors in the configured collection."""
        return int(self.collection.count())

    def iter_batches(self) -> Iterator[list[BackupRecord]]:
        """Yield ChromaDB points as generic backup records."""
        offset = 0

        while True:
            response = self.collection.get(
                limit=self.config.batch_size,
                offset=offset,
                include=["embeddings", "metadatas"],
            )
            ids = response.get("ids", [])
            if not ids:
                return

            embeddings = cast(list[list[float]] | None, response.get("embeddings"))
            metadatas = cast(list[dict[str, Any]] | None, response.get("metadatas"))
            if embeddings is None or metadatas is None:
                raise ValueError(
                    "ChromaDB backup requires embeddings and metadatas in collection.get."
                )

            batch: list[BackupRecord] = []
            for record_id, vector, metadata in zip(ids, embeddings, metadatas, strict=True):
                payload_json = cast(str | None, metadata.get("payload_json"))
                if not payload_json:
                    raise ValueError(
                        f"ChromaDB source record '{record_id}' is missing payload_json metadata."
                    )
                batch.append(
                    BackupRecord(
                        id=str(record_id),
                        vector=[float(value) for value in vector],
                        payload_json=payload_json,
                    )
                )

            yield batch
            offset += len(ids)


class PineconeBackupSource:
    """Pinecone backup adapter used for namespace-level exports."""

    def __init__(self, config: BackupConfig) -> None:
        if not config.vector_store_api_key:
            raise ValueError("Pinecone backup requires PINECONE_API_KEY or VECTOR_STORE_API_KEY.")
        if not config.pinecone_index_name:
            raise ValueError("Pinecone backup requires PINECONE_INDEX_NAME.")

        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise RuntimeError(
                "Pinecone backup requires the optional 'pinecone' dependency."
            ) from exc

        self.config = config
        self.client = Pinecone(api_key=config.vector_store_api_key)
        self.index_host = config.pinecone_index_host or config.vector_store_url
        self.index_name = config.pinecone_index_name
        self.namespace = config.collection_name
        self.index = self._get_index()

    def count_points(self) -> int:
        """Return the total number of Pinecone vectors in the configured namespace."""
        stats = self.index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
        namespace_stats = namespaces.get(self.namespace)
        if namespace_stats is None:
            return 0
        return int(
            getattr(namespace_stats, "vector_count", None) or namespace_stats.get("vector_count", 0)
        )

    def iter_batches(self) -> Iterator[list[BackupRecord]]:
        """Yield Pinecone vectors as generic backup records."""
        for ids in _chunked(self._list_ids(), self.config.batch_size):
            fetched = self.index.fetch(ids=ids, namespace=self.namespace)
            vectors = getattr(fetched, "vectors", None) or fetched.get("vectors", {})
            batch: list[BackupRecord] = []
            for record_id in ids:
                vector_record = vectors.get(record_id)
                if vector_record is None:
                    raise ValueError(
                        f"Pinecone source record '{record_id}' was not returned by fetch."
                    )

                values = getattr(vector_record, "values", None) or vector_record.get("values")
                metadata = getattr(vector_record, "metadata", None) or vector_record.get("metadata")
                batch.append(
                    BackupRecord(
                        id=record_id,
                        vector=[float(value) for value in cast(list[float], values)],
                        payload_json=_serialize_payload(metadata, record_id),
                    )
                )

            if batch:
                yield batch

    def _get_index(self) -> Any:
        if self.index_host:
            return self.client.Index(host=self.index_host)

        description = self.client.describe_index(name=self.index_name)
        host = getattr(description, "host", None) or description.get("host")
        if not host:
            raise ValueError(f"Unable to resolve Pinecone host for index '{self.index_name}'.")
        return self.client.Index(host=host)

    def _list_ids(self) -> list[str]:
        list_paginated = getattr(self.index, "list_paginated", None)
        if callable(list_paginated):
            return self._list_ids_paginated(list_paginated)

        list_vectors = getattr(self.index, "list", None)
        if callable(list_vectors):
            return _flatten_pinecone_ids(list_vectors(namespace=self.namespace))

        raise ValueError("Pinecone backup requires an index.list or index.list_paginated API.")

    def _list_ids_paginated(self, list_paginated: Any) -> list[str]:
        collected: list[str] = []
        pagination_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "namespace": self.namespace,
                "limit": self.config.batch_size,
            }
            if pagination_token:
                kwargs["pagination_token"] = pagination_token
            response = list_paginated(**kwargs)
            collected.extend(_flatten_pinecone_ids(response))
            pagination_token = _extract_pinecone_pagination_token(response)
            if not pagination_token:
                return collected


class PGVectorBackupSource:
    """PGVector backup adapter used for PostgreSQL-backed collections."""

    def __init__(self, config: BackupConfig) -> None:
        if not config.pgvector_dsn:
            raise ValueError("PGVector backup requires PGVECTOR_DSN.")

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError(
                "PGVector backup requires the optional 'psycopg' and 'pgvector' dependencies."
            ) from exc

        self._psycopg = psycopg
        self._register_vector = register_vector
        self.config = config
        self.table_name = sanitize_collection_token(
            f"{config.pgvector_schema}_{config.collection_name}"
        )

    def count_points(self) -> int:
        """Return the total number of PGVector rows in the configured table."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM "{self.table_name}"')
            row = cursor.fetchone()
            return int(row[0] if row else 0)

    def iter_batches(self) -> Iterator[list[BackupRecord]]:
        """Yield PGVector rows as generic backup records."""
        offset = 0

        while True:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    f'''
                    SELECT id, payload, embedding
                    FROM "{self.table_name}"
                    ORDER BY id
                    LIMIT %s OFFSET %s
                    ''',
                    (self.config.batch_size, offset),
                )
                rows = cursor.fetchall()

            if not rows:
                return

            batch = [
                BackupRecord(
                    id=str(record_id),
                    vector=[float(value) for value in cast(list[float], vector)],
                    payload_json=_serialize_payload(_coerce_pg_payload(payload), record_id),
                )
                for record_id, payload, vector in rows
            ]
            yield batch
            offset += len(rows)

    def _connect(self) -> Any:
        connection = self._psycopg.connect(self.config.pgvector_dsn)
        self._register_vector(connection)
        return connection


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
    manifest_path = output_path / "backup-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "vector_store": config.vector_store,
                "backup_format": "chromadb",
                "collection_name": config.collection_name,
                "embedding_provider": config.embedding_provider,
                "embedding_model": config.embedding_model,
                "embedding_dimension": config.embedding_dimension,
                "output_path": str(output_path),
                "remote_points": remote_points,
                "local_points": local_points,
                "artifact_size_bytes": artifact_size_bytes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
        manifest_path=manifest_path,
    )


def build_backup_source(config: BackupConfig) -> VectorStoreBackupSource:
    """Resolve the script-local vector store adapter from the provided config."""
    if config.vector_store == "qdrant":
        return QdrantBackupSource(config)
    if config.vector_store == "chromadb":
        return ChromaDBBackupSource(config)
    if config.vector_store == "pinecone":
        return PineconeBackupSource(config)
    if config.vector_store == "pgvector":
        return PGVectorBackupSource(config)

    raise ValueError(f"Unsupported vector store for backup: {config.vector_store}")


def load_config(argv: list[str] | None = None) -> BackupConfig:
    """Resolve script configuration from CLI args with CI-friendly env fallbacks."""
    vector_store = _env_or_default("VECTOR_STORE", "qdrant")
    ingestion_outputs = _read_ingestion_outputs_env()
    collection_name = (
        _env_or_none("BACKUP_COLLECTION_NAME")
        or _env_or_none("VECTOR_STORE_TARGET_NAME")
        or _env_or_none("QDRANT_COLLECTION_NAME")
        or ingestion_outputs.get("VECTOR_STORE_TARGET_NAME")
        or ingestion_outputs.get("QDRANT_COLLECTION_NAME")
        or "movies"
    )
    output_root = _env_or_default("BACKUP_OUTPUT_ROOT", str(BACKUP_ROOT))
    batch_size = int(_env_or_default("BACKUP_BATCH_SIZE", "1000"))
    vector_store_url = _env_or_none("VECTOR_STORE_URL") or _env_or_none("QDRANT_URL")
    vector_store_api_key = _env_or_none("VECTOR_STORE_API_KEY") or _env_or_none("QDRANT_API_KEY_RW")
    embedding_dimension = _env_or_none("EMBEDDING_DIMENSION") or ingestion_outputs.get(
        "EMBEDDING_DIMENSION"
    )
    chromadb_persist_path = _env_or_none("CHROMADB_PERSIST_PATH")
    pinecone_index_name = _env_or_none("PINECONE_INDEX_NAME")
    pinecone_index_host = _env_or_none("PINECONE_INDEX_HOST")
    pgvector_dsn = _env_or_none("PGVECTOR_DSN")
    pgvector_schema = _env_or_default("PGVECTOR_SCHEMA", "public")

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
    parser.add_argument(
        "--chromadb-persist-path",
        default=chromadb_persist_path,
        help="ChromaDB persistence path for chromadb source backups.",
    )
    parser.add_argument(
        "--pinecone-index-name",
        default=pinecone_index_name,
        help="Pinecone index name for pinecone source backups.",
    )
    parser.add_argument(
        "--pinecone-index-host",
        default=pinecone_index_host,
        help="Optional Pinecone index host override.",
    )
    parser.add_argument(
        "--pgvector-dsn",
        default=pgvector_dsn,
        help="PGVector PostgreSQL DSN for pgvector source backups.",
    )
    parser.add_argument(
        "--pgvector-schema",
        default=pgvector_schema,
        help="PGVector schema token used when deriving table names.",
    )

    args = parser.parse_args(argv)
    return BackupConfig(
        vector_store=args.vector_store,
        collection_name=args.collection_name,
        output_root=Path(args.output_root),
        batch_size=args.batch_size,
        embedding_provider=_env_or_none("EMBEDDING_PROVIDER")
        or ingestion_outputs.get("EMBEDDING_PROVIDER"),
        embedding_model=_env_or_none("EMBEDDING_MODEL") or ingestion_outputs.get("EMBEDDING_MODEL"),
        embedding_dimension=int(embedding_dimension) if embedding_dimension else None,
        vector_store_url=args.vector_store_url,
        vector_store_api_key=args.vector_store_api_key,
        chromadb_persist_path=args.chromadb_persist_path,
        pinecone_index_name=args.pinecone_index_name,
        pinecone_index_host=args.pinecone_index_host,
        pgvector_dsn=args.pgvector_dsn,
        pgvector_schema=args.pgvector_schema,
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


def _read_ingestion_outputs_env() -> dict[str, str]:
    """Read ingestion outputs emitted by the pipeline when present."""
    if not INGESTION_OUTPUTS_PATH.exists():
        return {}

    parsed: dict[str, str] = {}
    for line in INGESTION_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        parsed[key] = value
    return parsed


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


def _coerce_pg_payload(payload: Any) -> dict[str, Any]:
    """Normalize PGVector JSON payload values into dictionaries."""
    if isinstance(payload, str):
        return cast(dict[str, Any], json.loads(payload))
    return cast(dict[str, Any], dict(payload))


def _chunked(values: list[str], size: int) -> Iterator[list[str]]:
    """Yield fixed-size slices from a list."""
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _flatten_pinecone_ids(value: Any) -> list[str]:
    """Collect vector IDs from Pinecone list-style responses."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        if "id" in value and isinstance(value["id"], str):
            return [value["id"]]
        if "ids" in value:
            return _flatten_pinecone_ids(value["ids"])
        if "vectors" in value:
            return _flatten_pinecone_ids(value["vectors"])
        if "matches" in value:
            return _flatten_pinecone_ids(value["matches"])
        collected: list[str] = []
        for key in ("vectors", "matches", "data", "results"):
            nested = value.get(key)
            if nested is None:
                continue
            collected.extend(_flatten_pinecone_ids(nested))
        return collected
    if isinstance(value, list | tuple):
        collected = []
        for nested in value:
            collected.extend(_flatten_pinecone_ids(nested))
        return collected
    if hasattr(value, "id") and isinstance(value.id, str):
        return [value.id]
    if hasattr(value, "__dict__"):
        return _flatten_pinecone_ids(vars(value))
    return []


def _extract_pinecone_pagination_token(response: Any) -> str | None:
    """Extract a pagination token from a Pinecone paginated list response."""
    pagination = getattr(response, "pagination", None)
    if pagination is None and isinstance(response, Mapping):
        pagination = response.get("pagination")
    if pagination is None:
        return None

    if isinstance(pagination, Mapping):
        token = pagination.get("next") or pagination.get("next_token") or pagination.get("token")
        return cast(str | None, token)

    return cast(
        str | None,
        getattr(pagination, "next", None)
        or getattr(pagination, "next_token", None)
        or getattr(pagination, "token", None),
    )


if __name__ == "__main__":
    backup()
