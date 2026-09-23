"""Vercel entrypoint: export the FastAPI application from main.py."""
from main import app

__all__ = ["app"]
