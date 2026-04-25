import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_backup_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "backup_vectorstore",
        Path(__file__).resolve().parents[1] / "scripts" / "backup_vectorstore.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/backup_vectorstore.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backup_script = _load_backup_script()


def test_build_backup_source_supports_all_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qdrant = object()
    chromadb = object()
    pinecone = object()
    pgvector = object()

    monkeypatch.setattr(backup_script, "QdrantBackupSource", lambda config: qdrant)
    monkeypatch.setattr(backup_script, "ChromaDBBackupSource", lambda config: chromadb)
    monkeypatch.setattr(backup_script, "PineconeBackupSource", lambda config: pinecone)
    monkeypatch.setattr(backup_script, "PGVectorBackupSource", lambda config: pgvector)

    assert (
        backup_script.build_backup_source(
            backup_script.BackupConfig(
                vector_store="qdrant",
                collection_name="movies_test_1",
                output_root=Path("outputs/backups"),
                batch_size=100,
            )
        )
        is qdrant
    )
    assert (
        backup_script.build_backup_source(
            backup_script.BackupConfig(
                vector_store="chromadb",
                collection_name="movies_test_1",
                output_root=Path("outputs/backups"),
                batch_size=100,
            )
        )
        is chromadb
    )
    assert (
        backup_script.build_backup_source(
            backup_script.BackupConfig(
                vector_store="pinecone",
                collection_name="movies_test_1",
                output_root=Path("outputs/backups"),
                batch_size=100,
            )
        )
        is pinecone
    )
    assert (
        backup_script.build_backup_source(
            backup_script.BackupConfig(
                vector_store="pgvector",
                collection_name="movies_test_1",
                output_root=Path("outputs/backups"),
                batch_size=100,
            )
        )
        is pgvector
    )


def test_load_config_prefers_generic_target_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs_file = tmp_path / "ingestion-outputs.env"
    outputs_file.write_text(
        "\n".join(
            [
                "VECTOR_STORE_TARGET_NAME=movies_baai_bge_m3_1024",
                "EMBEDDING_PROVIDER=huggingface",
                "EMBEDDING_MODEL=BAAI/bge-m3",
                "EMBEDDING_DIMENSION=1024",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(backup_script, "INGESTION_OUTPUTS_PATH", outputs_file)
    monkeypatch.delenv("BACKUP_COLLECTION_NAME", raising=False)
    monkeypatch.delenv("VECTOR_STORE_TARGET_NAME", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)

    config = backup_script.load_config([])

    assert config.collection_name == "movies_baai_bge_m3_1024"
    assert config.embedding_provider == "huggingface"
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.embedding_dimension == 1024
