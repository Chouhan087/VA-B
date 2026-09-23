# Move VOXIA AI database to Supabase

## 1. Create the Supabase project
- Create a project at https://supabase.com/dashboard.
- Save the database password securely.
- In **Database -> Extensions**, enable `vector` (pgvector).
- From **Connect**, copy the **Session pooler** PostgreSQL connection string for serverless use.

## 2. Configure Vercel backend environment
Add these variables to the backend Vercel project:
- `DATABASE_URL`: the Supabase Session pooler URL. Preserve its username/host/port and password exactly; include `sslmode=require`.
- `EMBEDDING_DIM=768`
- `VECTOR_INDEX_METHOD=hnsw` (or `none` if you want to skip ANN index creation)
- Existing OpenRouter, JWT, CORS, frontend, and OAuth environment variables.

Do not place database credentials in the frontend or commit them to Git.

## 3. Deploy
The backend uses SQLAlchemy models in `models_db.py`. On startup, `main.py` runs `Base.metadata.create_all()` and sets up pgvector indexes. This creates missing tables; it is not a schema migration system for future model changes. Use Alembic or explicit SQL migrations when altering an already deployed schema.

## 4. Migrate existing local data (optional)
Back up `backend/voxia.db` first. Then, from the backend folder, set `DATABASE_URL` to your Supabase URL and run:

```powershell
python migrate_sqlite_to_postgres.py .\voxia.db
```

The script copies users, conversations, messages, documents, document chunks, memories, and reminders in foreign-key order. It does not delete the SQLite file. Do not run repeatedly against a populated destination without clearing/reconciling duplicate rows first.

## 5. Verify
- Check the backend `/api/health` endpoint.
- Create a test account, send a chat, create a reminder, and upload a small text-based PDF.
- Restart/redeploy and confirm the records remain available.
- Confirm RAG retrieval works after enabling pgvector.

## Local development
If `DATABASE_URL` is not set, VOXIA continues using SQLite (`sqlite:///./voxia.db`).
