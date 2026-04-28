import json
from pathlib import Path

INGESTION_OUTPUTS_PATH = Path("ingestion-outputs.env")
REPORT_PATH = Path("outputs/reports/cost-report.json")


def main() -> None:
    """Create or refresh a CI-readable cost report from ingestion outputs."""
    values = _read_env_file(INGESTION_OUTPUTS_PATH)
    report = {
        "embedding_provider": values.get("EMBEDDING_PROVIDER"),
        "embedding_model": values.get("EMBEDDING_MODEL"),
        "embedding_dimension": _to_int(values.get("EMBEDDING_DIMENSION")),
        "vector_store": values.get("VECTOR_STORE"),
        "target_name": values.get("VECTOR_STORE_TARGET_NAME"),
        "collection_name": values.get("VECTOR_COLLECTION_PREFIX"),
        "prompt_tokens": _to_int(values.get("INGESTION_PROMPT_TOKENS")),
        "total_tokens": _to_int(values.get("INGESTION_TOTAL_TOKENS")),
        "estimated_cost_usd": _to_float(values.get("INGESTION_ESTIMATED_COST_USD")),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run ingestion before cost reporting.")

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value
    return values


def _to_int(value: str | None) -> int:
    return int(value) if value else 0


def _to_float(value: str | None) -> float:
    return float(value) if value else 0.0


if __name__ == "__main__":
    main()
