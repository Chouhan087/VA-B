"""
Database setup for VOXIA AI.

Default: SQLite, single file `voxia.db` next to this module — zero setup,
matches the project's free-first / low-friction ethos.

To move to PostgreSQL (with pgvector for real vector search at scale),
set DATABASE_URL to a postgres:// URL, e.g.:
    DATABASE_URL=postgresql://voxia:password@localhost:5432/voxia
and make sure the `vector` extension is installed on that Postgres server
(Ubuntu/Debian: `apt install postgresql-16-pgvector` for a matching
version, then this app creates the extension itself on startup).
That's it — models_db.py's EmbeddingType picks the right column type per
backend automatically, and rag.py picks the right search strategy.

EMBEDDING_DIM: pgvector columns need a fixed vector dimension (SQLite's
JSON-blob storage doesn't care). This must match whatever embedding model
actually produces your vectors — the default (768) matches Ollama's
nomic-embed-text, which is also this project's recommended embedding
model (see backend/.env.example). If you switch to a model with a
different output dimension, update EMBEDDING_DIM to match *and* re-embed
existing documents/memories (old vectors won't fit the new column, and
wouldn't be comparable to the new model's vectors anyway).

VECTOR_INDEX_METHOD: without an index, rag.py's Postgres path (see
find_relevant()) still works correctly — pgvector's cosine_distance still
runs — it's just an exact sequential scan, fine up to maybe tens of
thousands of rows and slower beyond that. Set this to build a real
approximate-nearest-neighbor index instead:
  - "hnsw" (default): best default for most cases — good recall, doesn't
    need a row-count estimate up front, handles ongoing inserts (new
    documents/memories) reasonably well without a manual rebuild.
  - "ivfflat": lower build cost, but needs the `lists` parameter roughly
    tuned to table size (this uses a rough sqrt(row_count) heuristic
    at creation time) and should be rebuilt (`REINDEX`) after the table
    has grown substantially, since its clusters are fixed at build time.
  - "none": skip indexing, exact search only (the original behavior).
"""

import math
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./voxia.db")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))
VECTOR_INDEX_METHOD = os.environ.get("VECTOR_INDEX_METHOD", "hnsw").lower()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

IS_POSTGRES = engine.dialect.name == "postgresql"

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def ensure_pgvector_extension():
    """
    Must run before Base.metadata.create_all() on Postgres, since the
    `vector` column type doesn't exist until this extension is created.
    No-op on SQLite. Requires the vector extension files to be installed
    on the Postgres server itself (this only activates it for this DB;
    it can't install the extension from scratch).
    """
    if not IS_POSTGRES:
        return
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def _table_row_count(conn, table_name: str) -> int:
    return conn.execute(text(f"SELECT count(*) FROM {table_name}")).scalar() or 0


def ensure_pgvector_indexes():
    """
    Must run AFTER Base.metadata.create_all() (the tables have to exist
    first). No-op on SQLite, and no-op if VECTOR_INDEX_METHOD=none. Safe
    to call every startup — CREATE INDEX IF NOT EXISTS / DROP-then-create
    guards make this idempotent rather than erroring or duplicating work.
    """
    if not IS_POSTGRES or VECTOR_INDEX_METHOD == "none":
        return

    with engine.connect() as conn:
        for table in ("document_chunks", "memories"):
            index_name = f"ix_{table}_embedding_{VECTOR_INDEX_METHOD}"

            if VECTOR_INDEX_METHOD == "hnsw":
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON {table} USING hnsw (embedding vector_cosine_ops)"
                    )
                )
            elif VECTOR_INDEX_METHOD == "ivfflat":
                # IVFFlat's cluster count should roughly track table size;
                # sqrt(n) is the commonly recommended starting point. Needs
                # a real REINDEX later if the table grows a lot afterward,
                # since clusters aren't recomputed automatically.
                row_count = _table_row_count(conn, table)
                lists = max(1, min(1000, round(math.sqrt(max(row_count, 1)))))
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON {table} USING ivfflat (embedding vector_cosine_ops) "
                        f"WITH (lists = {lists})"
                    )
                )
            else:
                raise ValueError(
                    f"Unknown VECTOR_INDEX_METHOD: {VECTOR_INDEX_METHOD!r} "
                    "(expected 'hnsw', 'ivfflat', or 'none')"
                )
        conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
