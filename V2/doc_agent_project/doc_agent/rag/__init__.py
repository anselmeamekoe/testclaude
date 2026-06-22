"""RAG subpackage: chunking, embeddings, and a FAISS vector index.

Public surface for building and querying a retrieval index over a repository.
"""

from .chunking import Chunk, chunk_repository
from .embeddings import (
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from .index import RetrievedChunk, VectorIndex

__all__ = [
    "Chunk",
    "chunk_repository",
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "RetrievedChunk",
    "VectorIndex",
]
