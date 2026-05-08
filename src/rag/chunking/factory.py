from rag.chunking.base import Chunker
from rag.chunking.field import FieldChunker
from rag.chunking.fixed_size import FixedSizeChunker
from rag.chunking.flat import FlatChunker
from rag.chunking.sentence import SentenceChunker
from rag.config import ChunkingStrategyName, settings


def create_chunker(strategy: ChunkingStrategyName) -> Chunker:
    if strategy == "flat":
        return FlatChunker()
    if strategy == "fixed_size":
        return FixedSizeChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    if strategy == "sentence":
        return SentenceChunker(
            min_sentences=settings.chunk_min_sentences,
            max_sentences=settings.chunk_max_sentences,
        )
    if strategy == "field":
        return FieldChunker(settings.chunk_fields_list)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def get_chunker() -> Chunker:
    return create_chunker(settings.chunking_strategy)
