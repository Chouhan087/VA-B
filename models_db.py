"""
ORM models for VOXIA AI persistence.

Mirrors the schema sketched in the project spec (section 20). MEMORY
(long-term memory) is now Memory; DOCUMENTS is Document + DocumentChunk.

Embedding columns (DocumentChunk.embedding, Memory.embedding) use
EmbeddingType, defined below: a real pgvector column on Postgres (enabling
fast, native ANN search — see rag.py), or a JSON-encoded text column on
SQLite (fine at MVP scale). Either way, the Python-level value is always
a plain list[float] — callers never need to know which backend is active.
"""

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None

from database import Base, EMBEDDING_DIM


class _JSONEmbeddingText(TypeDecorator):
    """
    Concrete storage for non-Postgres backends (SQLite): a plain
    JSON-encoded list[float] in a text column. Used as the "sqlite"
    variant of EmbeddingType below — application code never touches this
    directly.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else json.dumps(value)

    def process_result_value(self, value, dialect):
        return None if value is None else json.loads(value)


if Vector is not None:
    # On Postgres this column IS pgvector's Vector type — not wrapped —
    # so its native comparator (.cosine_distance(), .l2_distance(), etc.,
    # used by rag.py) works directly. with_variant swaps in the JSON/text
    # storage only for sqlite; every other dialect gets the real Vector.
    EmbeddingType = Vector(EMBEDDING_DIM).with_variant(_JSONEmbeddingText(), "sqlite")
else:
    # pgvector package not installed at all: fall back everywhere. Fine
    # for SQLite-only use; running against Postgres without the pgvector
    # package installed isn't a supported configuration (see requirements.txt).
    EmbeddingType = _JSONEmbeddingText()


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=True)  # null for Google-only accounts
    google_id = Column(String, unique=True, nullable=True, index=True)
    created_at = Column(DateTime, default=_now)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    reminders = relationship("Reminder", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=_now)

    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=_now)
    chunk_count = Column(Integer, default=0)

    user = relationship("User", back_populates="documents")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan", order_by="DocumentChunk.chunk_index"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    # Denormalized from Document.user_id: lets ownership-scoped vector
    # search (rag.find_relevant) filter with a plain WHERE clause instead
    # of a join, which matters for pushing the search down to pgvector.
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(EmbeddingType, nullable=False)

    document = relationship("Document", back_populates="chunks")


class Memory(Base):
    """
    A single durable fact extracted from something the user said (e.g.
    "Is preparing for a DBMS exam"), independent of any one conversation —
    this is what lets VOXIA recall context across separate chat sessions.
    """

    __tablename__ = "memories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(EmbeddingType, nullable=False)
    source_conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True)
    created_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="memories")


class Reminder(Base):
    """
    Backs the "reminders" tool (see tools.py) — a simple to-do/reminder
    list the AI agent can add to and read from via natural language, e.g.
    "remind me to submit the assignment" / "what are my reminders?".
    """

    __tablename__ = "reminders"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    done = Column(Integer, default=0)  # 0/1 — SQLite has no native bool
    created_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="reminders")
