"""
Copy data from VOXIA's local SQLite database into the configured PostgreSQL DB.

Usage from the backend directory:
    python migrate_sqlite_to_postgres.py .\voxia.db

Set DATABASE_URL to the Supabase Session pooler URL before running.
This script creates missing destination tables and copies rows in FK order.
Back up both databases first. It does not delete or modify the source SQLite file.
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, select, insert
from sqlalchemy.orm import Session

# Load .env before importing database/models.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)

from database import Base, engine as destination_engine, ensure_pgvector_extension
from models_db import User, Conversation, Message, Document, DocumentChunk, Memory, Reminder

MODELS_IN_FK_ORDER = [User, Conversation, Message, Document, DocumentChunk, Memory, Reminder]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite_path", help="Path to the source SQLite file")
    args = parser.parse_args()

    if destination_engine.dialect.name != "postgresql":
        sys.exit("DATABASE_URL must point to PostgreSQL/Supabase for this migration.")

    source_path = os.path.abspath(args.sqlite_path)
    if not os.path.isfile(source_path):
        sys.exit(f"SQLite database file not found: {source_path}")

    source_engine = create_engine(
        "sqlite:///" + source_path.replace("\\", "/"),
        connect_args={"check_same_thread": False},
    )

    ensure_pgvector_extension()
    Base.metadata.create_all(bind=destination_engine)

    copied = {}
    with Session(source_engine) as source, Session(destination_engine) as destination:
        try:
            for model in MODELS_IN_FK_ORDER:
                rows = source.execute(select(model)).scalars().all()
                if not rows:
                    copied[model.__tablename__] = 0
                    continue

                payload = []
                for row in rows:
                    record = {column.name: getattr(row, column.name) for column in model.__table__.columns}
                    payload.append(record)

                destination.execute(insert(model.__table__), payload)
                copied[model.__tablename__] = len(payload)

            destination.commit()
        except Exception:
            destination.rollback()
            raise

    source_engine.dispose()
    destination_engine.dispose()

    print("Migration completed. Rows copied:")
    for table, count in copied.items():
        print(f"  {table}: {count}")


if __name__ == "__main__":
    main()
