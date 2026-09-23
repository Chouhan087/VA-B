# Deploy VOXIA AI FastAPI backend to Vercel

## Files added for Vercel
- `index.py` exports `app` from `main.py`, giving Vercel a recognized entrypoint.
- `GET /` is a simple root/index route returning the API name, status, docs path, and health path.
- `vercel.json` applies a 120-second function duration to `index.py`.

## Dashboard deployment
1. Push this project to GitHub.
2. In Vercel, choose **Add New → Project** and import the repository.
3. Set **Root Directory** to `VOXIA_AI_NoKeyGuard/backend` (or the backend directory path relative to the repository).
4. Use automatic framework detection for FastAPI/Python. Do not set a frontend build command or output directory for this backend-only project.
5. Add backend environment variables in **Settings → Environment Variables**:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_BASE_URL`
   - `OPENROUTER_MODEL`
   - `DATABASE_URL` (Supabase PostgreSQL connection string, if using Supabase)
   - `CORS_ORIGINS` (include the exact deployed frontend origin)
   - `JWT_SECRET`
   - Any Google OAuth variables and other settings used by your backend.
6. Deploy. If you change environment variables, redeploy for the changes to take effect.

## Verify
After deployment, open:
- `https://YOUR-BACKEND-DOMAIN.vercel.app/` — root status JSON
- `https://YOUR-BACKEND-DOMAIN.vercel.app/api/health` — health endpoint
- `https://YOUR-BACKEND-DOMAIN.vercel.app/docs` — interactive API docs

## Notes
- Keep secrets out of Git and out of frontend environment variables.
- If deploying from a monorepo, ensure the configured Root Directory points to the directory containing `index.py`, `main.py`, and `requirements.txt`.
- The app uses Supabase only when `DATABASE_URL` is configured for PostgreSQL. Ensure pgvector is enabled if using document/memory vector search.
- Check Vercel function logs if the deployment fails to import dependencies or connect to Supabase.
