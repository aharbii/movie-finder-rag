import os

import chromadb
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# 1. Connect to your source (Qdrant) and destination (Chroma)
q_client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
)
c_client = chromadb.PersistentClient()

# 2. Setup Chroma Collection
# Note: Ensure the distance metric matches your Qdrant config (e.g., 'cosine')
chroma_coll = c_client.get_or_create_collection(
    name="movie_plots", metadata={"hnsw:space": "cosine"}
)

# 3. Pull from Qdrant and Push to Chroma
offset = None
while True:
    # Scroll retrieves batches of points including the 'vector'
    points, offset = q_client.scroll(
        collection_name="text-embedding-3-large",
        with_vectors=True,
        with_payload=True,
        limit=100,  # Adjust batch size based on memory
        offset=offset,
    )

    if not points:
        break

    # Prepare data for Chroma
    chroma_coll.add(
        ids=[str(p.id) for p in points],
        embeddings=[p.vector for p in points],
        metadatas=[p.payload for p in points],
        documents=[
            p.payload.get("plot", "") for p in points
        ],  # Optional: if you store text in payload
    )

    if offset is None:
        break

print("Migration complete! You can now run your RAG tests against the local Chroma DB.")
