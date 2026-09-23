"""
RAG engine for VOXIA AI.

Pipeline: PDF -> text extraction -> chunking -> embeddings -> stored via
EmbeddingType (models_db.py) -> similarity search at query time.

Embeddings: tries Ollama's /api/embeddings first (free, local, matches the
project's free-first architecture). If Ollama is unreachable, falls back
to a deterministic hashing-based bag-of-words vector — not a real
semantic embedding, but it does capture literal word overlap, so
retrieval still works reasonably for the "no LLM installed" demo path,
consistent with llm.py's mock fallback. Both are sized to EMBEDDING_DIM
(database.py) so they fit the same storage column.

Search strategy: on Postgres, find_relevant() pushes the nearest-neighbor
search down to the database via pgvector's cosine_distance operator
(fast, indexable, scales to large tables). On SQLite, it fetches the
candidate rows and computes cosine similarity in Python (fine at MVP
scale, not fine at scale — which is exactly why the Postgres path
exists). Callers don't need to know or care which path ran.
"""

import hashlib
import math
import io
import os
import re

import httpx
from pypdf import PdfReader
from sqlalchemy import select

from database import IS_POSTGRES, EMBEDDING_DIM

CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
TOP_K = 4
MIN_SIMILARITY = 0.10  # below this, a chunk isn't worth injecting as context


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
    "on", "at", "for", "with", "by", "from", "and", "or", "but", "if", "as", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "my", "your",
    "his", "her", "our", "their", "what", "which", "who", "whom", "when", "where", "why",
    "how", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might",
    "not", "no", "so", "than", "then", "there", "here", "about", "into", "over", "under",
    "again", "further", "have", "has", "had", "having", "get", "got", "just", "also", "some",
}


def _fallback_embedding(text: str) -> list:
    """
    Deterministic feature-hashed bag-of-words vector, stopwords excluded.
    No LLM required. Same content word -> same bucket every time, so
    cosine similarity between two chunks/queries reflects meaningful
    vocabulary overlap rather than shared stopwords. Sized to
    EMBEDDING_DIM so it fits the same storage column as real embeddings
    (see database.py's docstring on why that matters for pgvector).
    """
    vector = [0.0] * EMBEDDING_DIM
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2]
    for word in words:
        bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


async def get_embedding(text: str) -> dict:
    """Stable local hashed-term vector; OpenRouter chat is not an embedding API."""
    return {"vector": _fallback_embedding(text), "source": "local-hash"}


def cosine_similarity(a: list, b: list) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def find_relevant(
    db,
    model_cls,
    owner_filter,
    query: str,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    query_vector: list = None,
) -> list:
    """
    Similarity search against a SQLAlchemy model (DocumentChunk or
    Memory). owner_filter is the SQLAlchemy filter condition scoping rows
    to the current user (and, for DocumentChunk, joined to Document) —
    e.g. `Memory.user_id == user_id`.

    On Postgres, the nearest-neighbor search runs in the database via
    pgvector's cosine_distance operator (ORDER BY ... LIMIT top_k) — fast
    and indexable. On SQLite, all matching rows are fetched and scored in
    Python instead. Either way, returns (score, row) tuples, highest
    first, above min_similarity.

    Pass a precomputed query_vector to skip re-embedding the same query
    when calling this more than once per request (e.g. once for documents,
    once for memories).
    """
    if query_vector is None:
        query_vector = (await get_embedding(query))["vector"]

    if IS_POSTGRES:
        stmt = (
            select(model_cls)
            .where(owner_filter)
            .order_by(model_cls.embedding.cosine_distance(query_vector))
            .limit(top_k)
        )
        rows = db.execute(stmt).scalars().all()
        scored = []
        for row in rows:
            if row.embedding is None or len(row.embedding) != len(query_vector):
                continue
            score = cosine_similarity(query_vector, row.embedding)
            if score >= min_similarity:
                scored.append((score, row))
        return scored

    # SQLite (or any non-Postgres backend): fetch candidates, score in Python.
    rows = db.execute(select(model_cls).where(owner_filter)).scalars().all()
    scored = []
    for row in rows:
        if row.embedding is None or len(row.embedding) != len(query_vector):
            continue
        score = cosine_similarity(query_vector, row.embedding)
        if score >= min_similarity:
            scored.append((score, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]
