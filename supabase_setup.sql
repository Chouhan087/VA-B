-- Run in Supabase SQL Editor if the vector extension is not already enabled.
create extension if not exists vector;

-- VOXIA's application tables are created by SQLAlchemy from models_db.py
-- when the backend starts. Do not manually create duplicate tables here.
