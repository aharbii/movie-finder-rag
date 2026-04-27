import argparse
import html
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from qdrant_client import QdrantClient

from rag.config import EmbeddingProviderName, settings
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.factory import create_embedding_provider

REPORT_DIR = Path("outputs/reports")
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE: float | None = None


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    collection_name: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class EvalQuery:
    query_id: str
    text: str
    expected_titles: list[str]
    category: str


COLLECTIONS = [
    CollectionSpec(
        collection_name="movies_text_embedding_3_large_3072",
        provider="openai",
        model="text-embedding-3-large",
    ),
    CollectionSpec(
        collection_name="movies_nomic_embed_text_latest_768",
        provider="ollama",
        model="nomic-embed-text:latest",
    ),
    CollectionSpec(
        collection_name="movies_all_minilm_latest_384",
        provider="ollama",
        model="all-minilm:latest",
    ),
]

EVAL_QUERIES = [
    EvalQuery(
        "q001",
        "I am trying to remember an older adventure comedy where a high school kid is friends with a strange scientist. There is a car that becomes a time machine, the kid accidentally goes back to when his parents were teenagers, and he has to make sure they fall in love so he can exist.",
        ["Back to the Future"],
        "direct_title_plot",
    ),
    EvalQuery(
        "q002",
        "The movie I mean has a programmer or hacker who thinks something is wrong with the world. People wear long black coats, there are agents chasing them, and eventually he learns everyday life is actually a computer-generated prison.",
        ["The Matrix"],
        "direct_title_plot",
    ),
    EvalQuery(
        "q003",
        "It is an animated movie from the point of view of children's toys. The cowboy toy gets jealous because a shiny astronaut action figure shows up and the kid starts liking him more.",
        ["Toy Story"],
        "direct_title_plot",
    ),
    EvalQuery(
        "q004",
        "I remember a rich man builds a theme park on an island with real dinosaurs. During a preview trip the security system fails, the electric fences stop working, and people have to survive a T rex and velociraptors.",
        ["Jurassic Park"],
        "direct_title_plot",
    ),
    EvalQuery(
        "q005",
        "This is the boxing film where a working class fighter from Philadelphia gets an unlikely chance to fight the heavyweight champion. There is training, running up steps, and the point is proving he can go the distance.",
        ["Rocky"],
        "direct_title_plot",
    ),
    EvalQuery(
        "q006",
        "I am looking for the famous mafia family movie. The father is a powerful crime boss, his sons are pulled into the family business, and the quiet younger son slowly becomes the new head of the organization.",
        ["The Godfather"],
        "genre_cast_hints",
    ),
    EvalQuery(
        "q007",
        "A boy who lives with terrible relatives finds out he is a wizard. He goes to a magical boarding school, makes friends, plays a flying sport, and learns about the dark wizard connected to his parents' death.",
        ["Harry Potter and the Philosopher's Stone"],
        "genre_cast_hints",
    ),
    EvalQuery(
        "q008",
        "This one is confusing because the characters keep going into dreams inside other dreams. A thief is hired to plant an idea in someone's mind, and there are rules about waking up, time slowing down, and a spinning top.",
        ["Inception"],
        "genre_cast_hints",
    ),
    EvalQuery(
        "q009",
        "It is almost entirely people arguing in a jury room. Everyone thinks the accused young man is guilty except one juror, and the discussion slowly exposes doubts in the evidence.",
        ["12 Angry Men"],
        "genre_cast_hints",
    ),
    EvalQuery(
        "q010",
        "A family goes to take care of a huge empty hotel during winter. The father is a writer, starts losing his mind, there is a little boy with psychic visions, and the hotel itself feels haunted.",
        ["The Shining"],
        "genre_cast_hints",
    ),
    EvalQuery(
        "q011",
        "I only remember a cute rusty robot left alone cleaning garbage on Earth after humans leave. Another sleek robot arrives, they barely talk, and the story becomes about humans in space and a plant.",
        ["WALL-E"],
        "vague_partial",
    ),
    EvalQuery(
        "q012",
        "The film follows a musician during World War II. He is separated from his family and survives by hiding in damaged buildings, and there is a scene where he plays piano for a German officer.",
        ["The Pianist"],
        "vague_partial",
    ),
    EvalQuery(
        "q013",
        "There is a young guy in Boston who is secretly brilliant at math but works as a janitor. A professor discovers him, but the emotional center is his therapy sessions with a counselor who challenges him.",
        ["Good Will Hunting"],
        "vague_partial",
    ),
    EvalQuery(
        "q014",
        "I remember an astronaut gets left behind because the crew thinks he died. He has to survive alone on Mars, figures out how to grow food with potatoes, and NASA tries to bring him back.",
        ["The Martian"],
        "vague_partial",
    ),
    EvalQuery(
        "q015",
        "A young drummer at a serious music school is pushed by a teacher who screams at him, throws things, and demands impossible tempo perfection. The ending is a long intense jazz performance.",
        ["Whiplash"],
        "vague_partial",
    ),
    EvalQuery(
        "q016",
        "The main character cannot leave his apartment and watches neighbors through the window. He starts thinking one neighbor killed his wife, but most of the movie is just him observing from across the courtyard.",
        ["Rear Window"],
        "adversarial_misleading",
    ),
    EvalQuery(
        "q017",
        "There is a woman who steals money and stops at a lonely motel. The owner seems awkward and controlled by his mother, and the famous shower scene happens before the story turns into a mystery about him.",
        ["Psycho"],
        "adversarial_misleading",
    ),
    EvalQuery(
        "q018",
        "I remember an action movie where a city bus has a bomb and cannot slow below a certain speed. A cop has to get on the bus while passengers panic and traffic keeps getting in the way.",
        ["Speed"],
        "adversarial_misleading",
    ),
    EvalQuery(
        "q019",
        "Animated Disney movie about two sisters. One of them has ice powers she cannot control, runs away to the mountains, and accidentally traps the kingdom in winter.",
        ["Frozen"],
        "adversarial_misleading",
    ),
    EvalQuery(
        "q020",
        "A Roman army commander is betrayed after the emperor dies. His family is killed, he becomes a slave, and later fights in arenas while trying to get revenge on the new ruler.",
        ["Gladiator"],
        "adversarial_misleading",
    ),
    EvalQuery(
        "q021",
        "The movie is about a man building a website at college that becomes huge. I remember lawsuits, arguments with friends and investors, and a cold feeling that the company success cost him his relationships.",
        ["The Social Network"],
        "long_memory",
    ),
    EvalQuery(
        "q022",
        "A suburban father becomes obsessed with changing his life after meeting his daughter's friend. There are roses, a strange neighbor filming things, and the story is about a family falling apart under the surface.",
        ["American Beauty"],
        "long_memory",
    ),
    EvalQuery(
        "q023",
        "Someone watches a cursed videotape, then the phone rings and says they have seven days. There is a creepy girl, a well, and an investigation into where the tape came from.",
        ["The Ring"],
        "long_memory",
    ),
    EvalQuery(
        "q024",
        "A weatherman keeps waking up to the exact same day in a small town. At first he uses it selfishly, then gets depressed, then slowly becomes better by learning things and helping people.",
        ["Groundhog Day"],
        "long_memory",
    ),
    EvalQuery(
        "q025",
        "A young con artist pretends to be a pilot, doctor, and lawyer while an FBI agent keeps chasing him. It has fake checks, disguises, and a cat-and-mouse feeling.",
        ["Catch Me If You Can"],
        "long_memory",
    ),
    EvalQuery(
        "q026",
        "The whole movie is harsh desert car chases. A woman drives a war rig, a captive group escapes a tyrant, and a mostly silent road warrior ends up helping them survive.",
        ["Mad Max: Fury Road"],
        "long_memory",
    ),
    EvalQuery(
        "q027",
        "A father clownfish crosses the ocean looking for his missing son. There is a forgetful blue fish helping him, turtles, jellyfish, and the son is trapped in a dentist's aquarium.",
        ["Finding Nemo"],
        "long_memory",
    ),
    EvalQuery(
        "q028",
        "A family secretly has superpowers but lives like normal people. The father misses being a hero, the kids have powers too, and eventually the whole family has to fight a villain.",
        ["The Incredibles"],
        "long_memory",
    ),
    EvalQuery(
        "q029",
        "A lonely man falls in love with a voice assistant or operating system. It is futuristic but quiet and emotional, more about intimacy and loneliness than robots fighting.",
        ["Her"],
        "long_memory",
    ),
    EvalQuery(
        "q030",
        "A detective solves a murder on a train stuck in snow. There are many passengers, everyone seems suspicious, and the solution involves more than one person being guilty.",
        ["Murder on the Orient Express"],
        "long_memory",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing Qdrant movie collections.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Optional cosine similarity cutoff for treating a retrieved expected movie as accepted.",
    )
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    parser.add_argument("--collection-name", default=None)
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama", "huggingface", "sentence-transformers", "google"),
        default=None,
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    specs = _resolve_specs(args.collection_name, args.provider, args.model)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key_rw)
    result = {
        "run_id": f"qdrant_live_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "timestamp_display": _format_timestamp(datetime.now(UTC)),
        "top_k": args.top_k,
        "min_score": args.min_score,
        "collections": [
            evaluate_collection(client, spec, args.top_k, args.min_score) for spec in specs
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = write_reports(result, output_dir)

    print(json.dumps(_summary(result), indent=2))
    for path in written_paths:
        print(f"Report artifact: {path}")


def evaluate_collection(
    client: QdrantClient,
    spec: CollectionSpec,
    top_k: int,
    min_score: float | None,
) -> dict[str, Any]:
    title_index = load_title_index(client, spec.collection_name)
    provider = build_provider(spec)
    per_query = []

    for query in EVAL_QUERIES:
        expected = [
            title_index[title.casefold()]
            for title in query.expected_titles
            if title.casefold() in title_index
        ]
        if not expected:
            per_query.append(
                {
                    "id": query.query_id,
                    "query": query.text,
                    "category": query.category,
                    "expected_titles": query.expected_titles,
                    "expected_ids": [],
                    "retrieved": [],
                    "precision_at_k": 0.0,
                    "recall_at_k": 0.0,
                    "reciprocal_rank": 0.0,
                    "target_rank": None,
                    "target_score": None,
                    "hit": False,
                    "accepted": False,
                    "skipped": True,
                }
            )
            continue

        vector = provider.embed(query.text)
        response = client.query_points(
            collection_name=spec.collection_name,
            query=vector,
            with_payload=True,
            limit=top_k,
        )
        retrieved = [
            {
                "id": int(point.payload["id"]),
                "title": str(point.payload["title"]),
                "release_year": int(point.payload["release_year"]),
                "plot": str(point.payload.get("plot", "")),
                "score": float(point.score),
            }
            for point in response.points
            if point.payload
        ]
        expected_ids = {movie["id"] for movie in expected}
        retrieved_ids = [movie["id"] for movie in retrieved]
        hits = expected_ids.intersection(retrieved_ids)
        target_rank = None
        target_score = None
        reciprocal_rank = 0.0
        for rank, movie in enumerate(retrieved, start=1):
            if movie["id"] in expected_ids:
                target_rank = rank
                target_score = cast(float, movie["score"])
                reciprocal_rank = 1.0 / rank
                break
        accepted = bool(hits) and (min_score is None or (target_score or 0.0) >= min_score)

        per_query.append(
            {
                "id": query.query_id,
                "query": query.text,
                "category": query.category,
                "expected_titles": [movie["title"] for movie in expected],
                "expected_ids": sorted(expected_ids),
                "retrieved": retrieved,
                "precision_at_k": len(hits) / top_k,
                "recall_at_k": len(hits) / len(expected_ids),
                "reciprocal_rank": reciprocal_rank,
                "target_rank": target_rank,
                "target_score": target_score,
                "hit": bool(hits),
                "accepted": accepted,
                "skipped": False,
            }
        )

    scored = [query for query in per_query if not query["skipped"]]
    usage = provider.get_model_usage()
    return {
        "collection_name": spec.collection_name,
        "provider": spec.provider,
        "model": spec.model,
        "point_count": client.count(collection_name=spec.collection_name, exact=True).count,
        "query_count": len(scored),
        "scores": {
            "precision_at_k": _mean(cast(float, query["precision_at_k"]) for query in scored),
            "recall_at_k": _mean(cast(float, query["recall_at_k"]) for query in scored),
            "mrr": _mean(cast(float, query["reciprocal_rank"]) for query in scored),
            "hit_rate_at_k": _mean(1.0 if query["hit"] else 0.0 for query in scored),
            "success_at_k": _mean(1.0 if query["accepted"] else 0.0 for query in scored),
            "mean_target_rank": _mean(
                float(cast(int, query["target_rank"]))
                for query in scored
                if query["target_rank"] is not None
            ),
            "mean_target_score": _mean(
                cast(float, query["target_score"])
                for query in scored
                if query["target_score"] is not None
            ),
        },
        "embedding_usage": usage.model_dump(),
        "per_query": per_query,
    }


def build_provider(spec: CollectionSpec) -> EmbeddingProvider:
    return create_embedding_provider(cast(EmbeddingProviderName, spec.provider), spec.model)


def _resolve_specs(
    collection_name: str | None,
    provider: str | None,
    model: str | None,
) -> list[CollectionSpec]:
    if collection_name is None:
        return COLLECTIONS
    if provider is None or model is None:
        raise ValueError("--provider and --model are required when --collection-name is provided.")
    return [CollectionSpec(collection_name=collection_name, provider=provider, model=model)]


def load_title_index(client: QdrantClient, collection_name: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    offset: Any = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            title = str(payload.get("title", ""))
            if title:
                index[title.casefold()] = {
                    "id": int(payload["id"]),
                    "title": title,
                }
        if offset is None:
            return index


def write_reports(result: dict[str, Any], output_dir: Path) -> list[Path]:
    """Write one report folder per collection plus a compact summary index."""
    written_paths: list[Path] = []
    summary_dir = output_dir / "qdrant-live-eval"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for collection in result["collections"]:
        collection_dir = summary_dir / _slug(str(collection["collection_name"]))
        collection_dir.mkdir(parents=True, exist_ok=True)
        collection_json = collection_dir / "report.json"
        collection_html = collection_dir / "index.html"
        collection_payload = {
            "run_id": result["run_id"],
            "timestamp_utc": result["timestamp_utc"],
            "timestamp_display": result["timestamp_display"],
            "top_k": result["top_k"],
            "min_score": result["min_score"],
            "collection": collection,
        }
        collection_json.write_text(
            json.dumps(collection_payload, indent=2) + "\n", encoding="utf-8"
        )
        collection_html.write_text(render_collection_html(collection_payload), encoding="utf-8")
        written_paths.extend([collection_json, collection_html])

    index_path = summary_dir / "index.html"
    summary_json = summary_dir / "summary.json"
    summary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(render_summary_html(result), encoding="utf-8")
    written_paths.extend([summary_json, index_path])
    return written_paths


def render_collection_html(result: dict[str, Any]) -> str:
    collection = result["collection"]
    scores = collection["scores"]
    css = """
    :root { --primary:#2563eb; --bg:#f1f5f9; --card:#fff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --dark:#1e293b; --success:#059669; --warn:#d97706; --danger:#dc2626; }
    body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; padding:2rem; background:var(--bg); color:var(--text); line-height:1.55; }
    .container { max-width:1400px; margin:0 auto; }
    header { display:flex; justify-content:space-between; gap:1rem; align-items:flex-end; margin-bottom:2rem; }
    h1 { margin:0; font-size:2.25rem; font-weight:800; color:var(--dark); }
    .meta { text-align:right; color:var(--muted); }
    .config { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1rem; background:var(--dark); padding:1.5rem; border-radius:1rem; margin-bottom:2.5rem; color:white; }
    .config-card { padding:.75rem 1rem; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1); border-radius:.75rem; }
    .label { display:block; font-size:.65rem; text-transform:uppercase; font-weight:700; color:#94a3b8; }
    .value { font-size:.85rem; font-weight:600; color:#f8fafc; }
    .tag { font-size:.65rem; background:var(--primary); padding:1px 4px; border-radius:4px; margin-left:6px; }
    .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:1.5rem; margin-bottom:2rem; }
    .stat { background:var(--card); border:1px solid var(--border); border-radius:1rem; padding:1.5rem; text-align:center; box-shadow:0 4px 6px -1px rgba(0,0,0,.05); }
    .stat-value { display:block; font-size:2.2rem; font-weight:800; }
    .stat-label { font-size:.8rem; font-weight:700; text-transform:uppercase; color:var(--muted); }
    .note { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; border-radius:.75rem; padding:1rem 1.25rem; margin-bottom:2rem; }
    .query-card { background:white; border:1px solid var(--border); border-radius:1rem; overflow:hidden; margin-bottom:1rem; }
    .query-head { padding:1.25rem 1.5rem; cursor:pointer; display:flex; gap:1rem; align-items:center; }
    .qid { font-family:monospace; font-weight:700; color:var(--muted); min-width:52px; }
    .qtext { flex:1; font-weight:600; }
    .scores { display:flex; gap:.5rem; flex-wrap:wrap; justify-content:flex-end; }
    .pill { font-family:monospace; font-size:.75rem; font-weight:800; padding:2px 6px; border-radius:4px; border:1px solid transparent; }
    .good { color:var(--success); } .good-bg { background:#ecfdf5; border-color:#a7f3d0; }
    .fair { color:var(--warn); } .fair-bg { background:#fffbeb; border-color:#fde68a; }
    .poor { color:var(--danger); } .poor-bg { background:#fef2f2; border-color:#fecaca; }
    .body { display:none; padding:1.5rem; border-top:1px solid var(--border); background:#fafafa; }
    .body.active { display:block; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }
    .box { background:white; border:1px solid var(--border); border-radius:.75rem; padding:1rem; }
    .gt { background:#fffbeb; border-color:#fde68a; color:#92400e; }
    table { width:100%; border-collapse:collapse; background:white; border-radius:.75rem; overflow:hidden; border:1px solid var(--border); }
    th, td { padding:.65rem .75rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; font-size:.88rem; }
    th { background:#f8fafc; color:#475569; font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }
    .plot { color:#475569; max-width:520px; }
    .matched { font-weight:800; color:var(--success); }
    .missed { font-weight:800; color:var(--danger); }
    @media (max-width:900px) { body{padding:1rem;} header{display:block;} .meta{text-align:left;margin-top:1rem;} .stats{grid-template-columns:1fr 1fr;} .query-head{display:block;} .grid{grid-template-columns:1fr;} table{display:block; overflow-x:auto;} }
    """
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="UTF-8"><title>Qdrant Collection Eval</title>',
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">',
        f'<style>{css}</style></head><body><div class="container">',
        "<header><div><h1>Qdrant Collection Evaluation</h1>",
        f'<div style="color:var(--muted); margin-top:.5rem;">Run ID: <code>{html.escape(result["run_id"])}</code></div></div>',
        f'<div class="meta"><b>{html.escape(result["timestamp_display"])}</b><br><code>{html.escape(result["timestamp_utc"])}</code></div></header>',
        '<div class="config">',
        _config("Collection", str(collection["collection_name"]), "qdrant"),
        _config("Embedding Model", str(collection["model"]), str(collection["provider"])),
        _config("Point Count", str(collection["point_count"]), "movies"),
        _config("Top K", str(result["top_k"]), "retrieval"),
        _config(
            "Similarity Cutoff",
            "None" if result["min_score"] is None else str(result["min_score"]),
            "cosine",
        ),
        _config("Eval Queries", str(collection["query_count"]), "curated"),
        "</div>",
        '<div class="note"><b>How to read this report:</b> Success@k is the primary gate for Movie Finder retrieval: the expected movie must appear in the returned candidate set. MRR and Target Rank show how high it appeared. Target Similarity is the Qdrant cosine similarity score for the expected movie when found. Precision@k is kept as a diagnostic because this app intentionally returns top-k candidates.</div>',
        '<div class="stats">',
        _stat("Success@k", scores["success_at_k"]),
        _stat("MRR", scores["mrr"]),
        _stat("Mean Target Rank", scores["mean_target_rank"], lower_is_better=True),
        _stat("Mean Target Similarity", scores["mean_target_score"]),
        "</div>",
        '<div class="stats">',
        _stat("Recall@k", scores["recall_at_k"]),
        _stat("Precision@k", scores["precision_at_k"], diagnostic=True),
        _stat("Hit Rate@k", scores["hit_rate_at_k"]),
        _stat(
            "Skipped Queries",
            float(len([q for q in collection["per_query"] if q["skipped"]])),
            diagnostic=True,
        ),
        "</div>",
    ]
    for query in collection["per_query"]:
        parts.append(_query_html(query))
    parts.extend(
        [
            "</div><script>",
            "function toggleCard(el){const b=el.nextElementSibling;b.classList.toggle('active');}",
            "</script></body></html>",
        ]
    )
    return "\n".join(parts)


def render_summary_html(result: dict[str, Any]) -> str:
    rows = []
    for collection in result["collections"]:
        scores = collection["scores"]
        slug = _slug(str(collection["collection_name"]))
        rows.append(
            "<tr>"
            f'<td><a href="{slug}/index.html">{html.escape(collection["collection_name"])}</a></td>'
            f"<td>{html.escape(collection['provider'])}</td>"
            f"<td>{html.escape(collection['model'])}</td>"
            f"<td>{collection['query_count']}</td>"
            f"<td>{scores['success_at_k']:.2f}</td>"
            f"<td>{scores['mrr']:.2f}</td>"
            f"<td>{scores['mean_target_rank']:.2f}</td>"
            f"<td>{scores['mean_target_score']:.4f}</td>"
            "</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qdrant Eval Summary</title>
<style>
body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f1f5f9; color:#0f172a; padding:2rem; }}
.container {{ max-width:1200px; margin:0 auto; }}
table {{ width:100%; border-collapse:collapse; background:white; border-radius:.75rem; overflow:hidden; border:1px solid #e2e8f0; }}
th,td {{ padding:.8rem 1rem; border-bottom:1px solid #e2e8f0; text-align:left; }}
th {{ background:#1e293b; color:white; font-size:.75rem; text-transform:uppercase; }}
a {{ color:#2563eb; font-weight:700; text-decoration:none; }}
</style>
</head>
<body>
<div class="container">
<h1>Qdrant Live Evaluation Summary</h1>
<p>Generated {html.escape(result["timestamp_display"])}. Each collection has its own HTML and JSON artifact folder.</p>
<table>
<thead><tr><th>Collection</th><th>Provider</th><th>Model</th><th>Queries</th><th>Success@k</th><th>MRR</th><th>Mean Rank</th><th>Mean Similarity</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
</body>
</html>"""


def _query_html(query: dict[str, Any]) -> str:
    retrieved = query["retrieved"]
    expected_ids = set(query.get("expected_ids", []))
    retrieved_items = "".join(
        "<tr>"
        f"<td>{rank}</td>"
        f"<td>{item['id']}</td>"
        f"<td>{html.escape(item['title'])}</td>"
        f"<td>{html.escape(str(item['release_year']))}</td>"
        f"<td>{item['score']:.4f}</td>"
        f"<td>{_match_label(item['id'] in expected_ids)}</td>"
        f'<td class="plot">{html.escape(_truncate(str(item["plot"]), 220))}</td>'
        "</tr>"
        for rank, item in enumerate(retrieved, start=1)
    )
    expected_items = "".join(f"{html.escape(title)}<br>" for title in query["expected_titles"])
    target_rank = query.get("target_rank")
    target_score = query.get("target_score")
    target_summary = (
        "Not found in top-k"
        if target_rank is None
        else f"Rank {target_rank}, cosine similarity {cast(float, target_score):.4f}"
    )
    precision = float(query["precision_at_k"])
    recall = float(query["recall_at_k"])
    mrr = float(query["reciprocal_rank"])
    success = 1.0 if query["accepted"] else 0.0
    skipped = bool(query["skipped"])
    return f"""
    <div class="query-card">
      <div class="query-head" onclick="toggleCard(this)">
        <span class="qid">{html.escape(query["id"])}</span>
        <span class="qtext">{html.escape(query["query"])}</span>
        <div class="scores">
          <span class="pill {_score_bg(success)}">Success: {bool(query["accepted"])}</span>
          <span class="pill {_score_bg(mrr)}">MRR: {mrr:.2f}</span>
          <span class="pill {_score_bg(recall)}">R: {recall:.2f}</span>
          <span class="pill {_score_bg(precision)}">P: {precision:.2f}</span>
        </div>
      </div>
      <div class="body">
        <div class="grid">
          <div class="box gt">
            <span class="label">Expected Movie</span>
            {expected_items}
            <div style="margin-top:.75rem;"><b>Target result:</b> {html.escape(target_summary)}</div>
            {'<div class="missed">Skipped because the expected title is absent from this collection.</div>' if skipped else ""}
          </div>
          <div class="box">
            <span class="label">Query Category</span>
            {html.escape(str(query["category"]).replace("_", " "))}
          </div>
        </div>
        <div style="margin-top:1.5rem;">
          <span class="label">Retrieved Top-K candidates</span>
          <table>
            <thead>
              <tr><th>Rank</th><th>Movie ID</th><th>Title</th><th>Year</th><th>Cosine Similarity</th><th>Expected?</th><th>Plot Snippet</th></tr>
            </thead>
            <tbody>{retrieved_items}</tbody>
          </table>
        </div>
      </div>
    </div>
    """


def _config(label: str, value: str, tag: str) -> str:
    return (
        '<div class="config-card">'
        f'<span class="label">{html.escape(label)}</span>'
        f'<span class="value">{html.escape(value)} <span class="tag">{html.escape(tag)}</span></span>'
        "</div>"
    )


def _stat(
    label: str,
    value: float,
    *,
    lower_is_better: bool = False,
    diagnostic: bool = False,
) -> str:
    score_value = 1.0 / value if lower_is_better and value else value
    return (
        '<div class="stat">'
        f'<span class="stat-value {_score_class(score_value)}">{value:.2f}</span>'
        f'<span class="stat-label">{html.escape(label)}{" (diagnostic)" if diagnostic else ""}</span>'
        "</div>"
    )


def _score_class(value: float) -> str:
    if value >= 0.8:
        return "good"
    if value >= 0.5:
        return "fair"
    return "poor"


def _score_bg(value: float) -> str:
    return f"{_score_class(value)}-bg"


def _match_label(value: bool) -> str:
    if value:
        return '<span class="matched">Expected</span>'
    return '<span style="color:#64748b;">Candidate</span>'


def _truncate(value: str, max_length: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "..."


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": result["run_id"],
        "top_k": result["top_k"],
        "collections": [
            {
                "collection_name": collection["collection_name"],
                "query_count": collection["query_count"],
                **collection["scores"],
                "embedding_usage": collection["embedding_usage"],
            }
            for collection in result["collections"]
        ],
    }


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%B %d, %Y at %H:%M:%S UTC")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "collection"


if __name__ == "__main__":
    main()
