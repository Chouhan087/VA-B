# Production image for the VOXIA AI backend.
#
# Build:  docker build -t voxia-backend ./backend
# Run:    docker run -p 8000:8000 --env-file .env voxia-backend
#
# Talks to SQLite (a file inside the container — fine for a demo, but its
# data disappears when the container is removed unless you mount a volume
# at /app) or Postgres (set DATABASE_URL — see docker-compose.yml for the
# recommended setup, which pairs this with a pgvector-enabled Postgres).

FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 (Postgres driver) — only needed if DATABASE_URL
# points at Postgres, but harmless (small) to always include so the same
# image works either way without a build-arg branch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 voxia \
    && mkdir -p /app/data \
    && chown -R voxia:voxia /app
USER voxia

# If left on the SQLite default, put the DB file somewhere explicit so a
# volume can be mounted at it (`-v voxia-data:/app/data`) to persist across
# container restarts/recreation.
ENV DATABASE_URL=sqlite:////app/data/voxia.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
