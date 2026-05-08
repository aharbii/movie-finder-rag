import argparse
import csv
import hashlib
import html
import json
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI

from evaluate_qdrant_collections import EVAL_QUERIES, EvalQuery
from rag.chunking.base import Chunker
from rag.chunking.field import FieldChunker
from rag.chunking.fixed_size import FixedSizeChunker
from rag.chunking.flat import FlatChunker
from rag.chunking.sentence import SentenceChunker
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.factory import get_embedding_provider
from rag.ingestion.csv_loader import load_movies
from rag.models.movie import Movie

DEFAULT_OUTPUT_PATH = Path("experiments/chunking-results.csv")
DEFAULT_REPORT_PATH = Path("experiments/chunking-report.html")
DEFAULT_CACHE_PATH = Path("outputs/cache/chunking-embeddings.json")
DEFAULT_CHROMADB_PATH = Path("outputs/chromadb/chunking-experiment")
DEFAULT_TOP_K = 5


@dataclass(frozen=True, slots=True)
class StrategySpec:
    name: str
    params: dict[str, Any]
    chunker: Chunker


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    movie: Movie
    chunk_text: str
    chunk_index: int
    vector: list[float]


@dataclass(frozen=True, slots=True)
class RankedMovie:
    movie: Movie
    score: float


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    top_k: int
    chromadb_path: Path
    output_path: Path
    report_path: Path


class EmbeddingCache:
    """Persistent text embedding cache for strategy sweeps."""

    def __init__(self, path: Path, provider: EmbeddingProvider) -> None:
        self.path = path
        self.provider = provider
        self._vectors = self._load()

    def embed(self, text: str) -> list[float]:
        key = _text_key(self.provider.model_info.name, text)
        cached = self._vectors.get(key)
        if cached is not None:
            return cached
        vector = self.provider.embed(text)
        self._vectors[key] = vector
        return vector

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._vectors), encoding="utf-8")

    def _load(self) -> dict[str, list[float]]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {str(key): [float(value) for value in values] for key, values in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunking strategy retrieval experiments.")
    parser.add_argument("--dataset", default="dataset/wiki_movie_plots_deduped.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--chromadb-path", default=str(DEFAULT_CHROMADB_PATH))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional corpus limit for local spike runs. Omit for the full supported dataset.",
    )
    args = parser.parse_args()

    provider = get_embedding_provider()
    cache = EmbeddingCache(Path(args.cache), provider)
    movies = load_movies(args.dataset)
    if args.limit is not None:
        movies = movies[: args.limit]

    config = ExperimentConfig(
        top_k=args.top_k,
        chromadb_path=Path(args.chromadb_path),
        output_path=Path(args.output),
        report_path=Path(args.report),
    )
    rows = run_experiment(movies, EVAL_QUERIES, strategy_matrix(), cache, config)
    cache.save()
    write_csv(rows, config.output_path)
    write_report(rows, config.report_path)
    print(f"Wrote {len(rows)} chunking experiment rows to {config.output_path}")
    print(f"Wrote chunking experiment report to {config.report_path}")


def run_experiment(
    movies: list[Movie],
    queries: list[EvalQuery],
    strategies: list[StrategySpec],
    cache: EmbeddingCache,
    config: ExperimentConfig,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    query_vectors = {query.query_id: cache.embed(query.text) for query in queries}
    client = chromadb.PersistentClient(path=str(config.chromadb_path))

    for strategy in strategies:
        chunks = embed_chunks(movies, strategy.chunker, cache)
        collection = rebuild_collection(client, strategy, chunks)
        per_query = [
            score_query(query, query_vectors[query.query_id], collection, config.top_k)
            for query in queries
        ]
        rows.append(
            {
                "strategy": strategy.name,
                "params": json.dumps(strategy.params, sort_keys=True),
                "movie_count": str(len(movies)),
                "chunk_count": str(len(chunks)),
                "top_k": str(config.top_k),
                "precision_at_k": f"{mean(row['precision_at_k'] for row in per_query):.6f}",
                "recall_at_k": f"{mean(row['recall_at_k'] for row in per_query):.6f}",
                "mrr": f"{mean(row['reciprocal_rank'] for row in per_query):.6f}",
                "hit_rate_at_k": f"{mean(row['hit'] for row in per_query):.6f}",
                "chromadb_path": str(config.chromadb_path),
                "collection_name": collection.name,
            }
        )

    return rows


def strategy_matrix() -> list[StrategySpec]:
    return [
        StrategySpec("flat", {}, FlatChunker()),
        StrategySpec("fixed_size", {"chunk_size": 120, "overlap": 24}, FixedSizeChunker(120, 24)),
        StrategySpec(
            "field",
            {"fields": ["plot", "cast", "metadata"]},
            FieldChunker(["plot", "cast", "metadata"]),
        ),
        StrategySpec(
            "sentence",
            {"min_sentences": 1, "max_sentences": 4},
            SentenceChunker(min_sentences=1, max_sentences=4),
        ),
    ]


def embed_chunks(
    movies: Iterable[Movie],
    chunker: Chunker,
    cache: EmbeddingCache,
) -> list[EmbeddedChunk]:
    chunks: list[EmbeddedChunk] = []
    for movie in movies:
        for chunk in chunker.chunk(movie):
            chunks.append(
                EmbeddedChunk(
                    movie=movie,
                    chunk_text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    vector=cache.embed(chunk.text),
                )
            )
    return chunks


def rebuild_collection(
    client: ClientAPI,
    strategy: StrategySpec,
    chunks: list[EmbeddedChunk],
) -> chromadb.Collection:
    collection_name = f"chunking_{strategy.name}_{_params_hash(strategy.params)}"
    with suppress(ValueError):
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "experiment": "chunking",
            "strategy": strategy.name,
            "params": json.dumps(strategy.params, sort_keys=True),
            "hnsw:space": "cosine",
        },
    )
    if not chunks:
        return collection

    collection.add(
        ids=[f"{chunk.movie.id}:{chunk.chunk_index}" for chunk in chunks],
        embeddings=cast(Any, [chunk.vector for chunk in chunks]),
        documents=[chunk.chunk_text for chunk in chunks],
        metadatas=[
            {
                "movie_id": chunk.movie.id,
                "title": chunk.movie.title,
                "release_year": chunk.movie.release_year,
                "chunk_index": chunk.chunk_index,
                "strategy": strategy.name,
            }
            for chunk in chunks
        ],
    )
    return collection


def score_query(
    query: EvalQuery,
    query_vector: list[float],
    collection: chromadb.Collection,
    top_k: int,
) -> dict[str, float]:
    ranked = rank_movies(collection, query_vector, top_k)
    expected_titles = {title.casefold() for title in query.expected_titles}
    retrieved_titles = [movie.movie.title.casefold() for movie in ranked]
    hits = expected_titles.intersection(retrieved_titles)
    reciprocal_rank = 0.0
    for rank, movie in enumerate(ranked, start=1):
        if movie.movie.title.casefold() in expected_titles:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision_at_k": len(hits) / top_k,
        "recall_at_k": len(hits) / len(expected_titles),
        "reciprocal_rank": reciprocal_rank,
        "hit": 1.0 if hits else 0.0,
    }


def rank_movies(
    collection: chromadb.Collection,
    query_vector: list[float],
    top_k: int,
) -> list[RankedMovie]:
    chunk_count = collection.count()
    if chunk_count == 0:
        return []

    result = collection.query(
        query_embeddings=cast(Any, [query_vector]),
        n_results=min(chunk_count, max(top_k * 10, top_k)),
        include=["distances", "metadatas"],
    )
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    best_by_movie_id: dict[int, RankedMovie] = {}
    for metadata, distance in zip(metadatas, distances, strict=True):
        if metadata is None:
            continue
        payload = cast(dict[str, object], metadata)
        movie_id = int(cast(str | int | float, payload["movie_id"]))
        movie = Movie(
            id=movie_id,
            title=str(payload["title"]),
            release_year=int(cast(str | int | float, payload["release_year"])),
            director="",
            genre=[],
            cast=[],
            plot="",
        )
        score = 1.0 - float(distance)
        current = best_by_movie_id.get(movie_id)
        if current is None or score > current.score:
            best_by_movie_id[movie_id] = RankedMovie(movie=movie, score=score)
    return sorted(best_by_movie_id.values(), key=lambda item: item.score, reverse=True)[:top_k]


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "params",
        "movie_count",
        "chunk_count",
        "top_k",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "hit_rate_at_k",
        "chromadb_path",
        "collection_name",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, str]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = max(rows, key=lambda row: float(row["mrr"]), default=None)
    baseline = next((row for row in rows if row["strategy"] == "flat"), None)
    delta = 0.0
    if best is not None and baseline is not None:
        baseline_mrr = float(baseline["mrr"])
        if baseline_mrr > 0.0:
            delta = (float(best["mrr"]) - baseline_mrr) / baseline_mrr

    body_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['strategy'])}</td>"
        f"<td><code>{html.escape(row['params'])}</code></td>"
        f"<td>{html.escape(row['chunk_count'])}</td>"
        f"<td>{float(row['precision_at_k']):.4f}</td>"
        f"<td>{float(row['recall_at_k']):.4f}</td>"
        f"<td>{float(row['mrr']):.4f}</td>"
        f"<td>{float(row['hit_rate_at_k']):.4f}</td>"
        f"<td><code>{html.escape(row['collection_name'])}</code></td>"
        "</tr>"
        for row in rows
    )
    recommendation = "No strategies were evaluated."
    if best is not None:
        recommendation = (
            f"Best strategy: {html.escape(best['strategy'])} with MRR {float(best['mrr']):.4f}. "
            f"MRR delta vs flat: {delta:.2%}."
        )
        if best["strategy"] == "field" and delta > 0.05:
            recommendation += " Field-based chunking meets the >5% adoption gate."
        elif best["strategy"] != "flat":
            recommendation += " Treat this as an experimental candidate; production adoption still requires chain retrieval coordination."
        else:
            recommendation += " Keep flat chunking unless a future run clears the adoption gate."

    report_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chunking Experiment Report</title>
<style>
body {{ font-family: Arial, sans-serif; color:#17202a; margin:2rem; }}
table {{ border-collapse: collapse; width:100%; }}
th, td {{ border:1px solid #d5d8dc; padding:.55rem; text-align:left; vertical-align:top; }}
th {{ background:#1f618d; color:white; }}
code {{ white-space:nowrap; }}
.summary {{ background:#f4f6f7; border:1px solid #d5d8dc; padding:1rem; margin-bottom:1rem; }}
</style>
</head>
<body>
<h1>Chunking Experiment Report</h1>
<div class="summary">
<p>{recommendation}</p>
<p>All strategies were evaluated against local ChromaDB collections with provider, corpus, query set,
and top-k held fixed. Only the chunking strategy and its parameters vary.</p>
</div>
<table>
<thead>
<tr><th>Strategy</th><th>Params</th><th>Chunks</th><th>Precision@k</th><th>Recall@k</th><th>MRR</th><th>Hit Rate@k</th><th>Local Collection</th></tr>
</thead>
<tbody>
{body_rows}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _text_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{digest}"


def _params_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":  # pragma: no cover
    main()
