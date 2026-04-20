import json
from pathlib import Path

from rag.config import settings
from rag.embeddings.factory import get_embedding_provider
from rag.vectorstore.factory import get_vector_store

REPORT_PATH = Path("outputs/reports/validation-report.json")


def validate() -> None:
    """Run a post-ingest smoke validation and persist a machine-readable report."""
    provider = get_embedding_provider()
    vector_store = get_vector_store()
    model_info = provider.model_info
    target_name = vector_store.target_name(model_info)
    point_count = vector_store.count(model_info)

    query_vector = provider.embed(settings.validation_query)
    results = []
    if query_vector:
        results = vector_store.search(query_vector, top_k=3, embedding_model=model_info)

    status = "passed" if point_count > 0 and bool(results) else "failed"
    report = {
        "status": status,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": model_info.name,
        "embedding_dimension": model_info.dimension,
        "vector_store": settings.vector_store,
        "target_name": target_name,
        "point_count": point_count,
        "validation_query": settings.validation_query,
        "result_count": len(results),
        "top_result": results[0].title if results else None,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if status != "passed":
        raise SystemExit(
            f"Validation failed for target '{target_name}'. "
            f"point_count={point_count}, result_count={len(results)}."
        )


if __name__ == "__main__":
    validate()
