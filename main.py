"""
VOXIA AI — FastAPI backend

Endpoints:
  GET  /api/health                     -> liveness + whether OpenRouter is configured
  GET  /api/roles                      -> list available assistant roles (public)

  POST /api/auth/signup                -> email+password signup, returns JWT
  POST /api/auth/login                 -> email+password login, returns JWT
  GET  /api/auth/me                    -> current user (requires JWT)
  GET  /api/auth/google/login          -> redirects to Google's consent screen
  GET  /api/auth/google/callback       -> Google redirects here; we issue our
                                           own JWT and redirect to the frontend

  POST /api/conversations              -> start a conversation (requires JWT)
  GET  /api/conversations              -> list the current user's conversations
  GET  /api/conversations/{id}         -> fetch one conversation + its messages
  POST /api/chat                       -> send a message, get a reply, STREAMED
                                           as Server-Sent Events (text/event-stream).
                                           Auto-grounds in documents, long-term
                                           memory, and tool results (calculator,
                                           reminders, weather) when relevant.

  POST /api/documents/upload           -> upload a PDF for RAG
  GET  /api/documents                  -> list the current user's documents
  DELETE /api/documents/{id}           -> delete a document

  GET  /api/memories                   -> list facts remembered about the user
  DELETE /api/memories/{id}            -> forget a specific fact

  GET  /api/reminders                  -> list reminders (also settable via chat,
  POST /api/reminders                     e.g. "remind me to...")
  PATCH /api/reminders/{id}/toggle
  DELETE /api/reminders/{id}

Storage: SQLite via SQLAlchemy (see database.py / models_db.py). Swappable
to PostgreSQL by setting DATABASE_URL — no other code changes needed.
"""

import asyncio
import json
import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load this project's backend/.env even when uvicorn is launched from another directory.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from roles import get_role, list_roles
import llm
from database import Base, engine, get_db, SessionLocal, ensure_pgvector_extension, ensure_pgvector_indexes
from models_db import User, Conversation, Message, Document, DocumentChunk, Memory, Reminder
from schemas import (
    SignupRequest,
    LoginRequest,
    TokenResponse,
    UserOut,
    NewConversationRequest,
    ChatRequest,
    ConversationOut,
    ConversationSummaryOut,
    DocumentOut,
    MemoryOut,
    ReminderOut,
)
import auth
import rag
import memory as memory_module
import tools

ensure_pgvector_extension()
Base.metadata.create_all(bind=engine)
ensure_pgvector_indexes()

app = FastAPI(title="VOXIA AI Backend", version="0.2.0")

# Wildcard by default (convenient for local dev / MVP use). For a public
# deployment, set CORS_ORIGINS to a comma-separated list of the frontend's
# actual origin(s), e.g. "https://yourdomain.com" — see DEPLOYMENT.md.
_cors_origins_env = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
)

_cors_origins = [
    o.strip().rstrip("/")
    for o in _cors_origins_env.split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- health / roles (public) ----------


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": "openrouter",
        "openrouter_configured": bool(llm.OPENROUTER_API_KEY),
        "model": llm.OPENROUTER_MODEL,
        "google_oauth_configured": auth.google_oauth_configured(),
    }

@app.get("/api/roles")
async def get_roles():
    return {"roles": list_roles()}


# ---------- auth ----------


@app.post("/api/auth/signup", response_model=TokenResponse)
async def signup(req: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user = User(email=req.email, name=req.name, password_hash=auth.hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(access_token=auth.create_access_token(user.id))


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if user is None or user.password_hash is None or not auth.verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    return TokenResponse(access_token=auth.create_access_token(user.id))


@app.get("/api/auth/me", response_model=UserOut)
async def me(current_user: User = Depends(auth.get_current_user)):
    return current_user


@app.get("/api/auth/google/login")
async def google_login():
    if not auth.google_oauth_configured():
        raise HTTPException(
            status_code=503,
            detail="Google login isn't configured on this server yet — set GOOGLE_CLIENT_ID / "
            "GOOGLE_CLIENT_SECRET (see backend/.env.example).",
        )
    state = auth.make_oauth_state()
    response = RedirectResponse(auth.build_google_auth_url(state))
    response.set_cookie("voxia_oauth_state", state, max_age=600, httponly=True,
                        secure=auth.FRONTEND_URL.startswith("https://"), samesite="lax")
    return response


@app.get("/api/auth/google/callback")
async def google_callback(code: str = None, state: str = None, error: str = None,
                          request: Request = None, db: Session = Depends(get_db)):
    # Validate state against a signed, short-lived cookie to prevent login CSRF.
    cookie_state = request.cookies.get("voxia_oauth_state") if request is not None else None
    if not state or not cookie_state or not secrets.compare_digest(state, cookie_state) or not auth.verify_oauth_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")
    if error or not code:
        response = RedirectResponse(f"{auth.FRONTEND_URL}/?auth_error=google_denied")
        response.delete_cookie("voxia_oauth_state")
        return response

    try:
        userinfo = await auth.exchange_code_for_userinfo(code)
    except Exception:
        return RedirectResponse(f"{auth.FRONTEND_URL}/?auth_error=google_exchange_failed")

    google_id = userinfo.get("sub")
    email = userinfo.get("email")
    name = userinfo.get("name") or (email.split("@")[0] if email else "VOXIA user")

    user = db.query(User).filter(User.google_id == google_id).first()
    if user is None and email:
        # Link to an existing email/password account if one exists, else create fresh.
        user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(email=email, name=name, google_id=google_id)
        db.add(user)
    else:
        user.google_id = user.google_id or google_id
    db.commit()
    db.refresh(user)

    token = auth.create_access_token(user.id)
    response = RedirectResponse(f"{auth.FRONTEND_URL}/?token={token}")
    response.delete_cookie("voxia_oauth_state")
    return response


# ---------- documents (RAG) ----------


MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


@app.post("/api/documents/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    safe_filename = os.path.basename((file.filename or "upload.pdf").replace("\\", "/"))[:180]
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (15 MB max)")
    if not raw.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file does not appear to be a PDF")

    try:
        text = rag.extract_text_from_pdf(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't read that PDF — is it a valid, non-encrypted file?")

    chunks = rag.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in that PDF (is it a scanned image?)")

    document = Document(user_id=current_user.id, filename=safe_filename, chunk_count=len(chunks))
    db.add(document)
    db.commit()
    db.refresh(document)

    for i, chunk_content in enumerate(chunks):
        embed_result = await rag.get_embedding(chunk_content)
        db.add(
            DocumentChunk(
                document_id=document.id,
                user_id=current_user.id,
                chunk_index=i,
                content=chunk_content,
                embedding=embed_result["vector"],
            )
        )
    db.commit()

    return document


@app.get("/api/documents", response_model=list[DocumentOut])
async def list_documents(current_user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )


@app.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None or document.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(document)
    db.commit()
    return {"deleted": True}


# ---------- long-term memory ----------


@app.get("/api/memories", response_model=list[MemoryOut])
async def list_memories(current_user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(Memory)
        .filter(Memory.user_id == current_user.id)
        .order_by(Memory.created_at.desc())
        .all()
    )


@app.delete("/api/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    mem = db.query(Memory).filter(Memory.id == memory_id).first()
    if mem is None or mem.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Memory not found")
    db.delete(mem)
    db.commit()
    return {"deleted": True}


# ---------- reminders (an AI agent tool, also manageable directly) ----------


class ReminderCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=500)


@app.get("/api/reminders", response_model=list[ReminderOut])
async def list_reminders(current_user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    reminders = (
        db.query(Reminder)
        .filter(Reminder.user_id == current_user.id)
        .order_by(Reminder.created_at.desc())
        .all()
    )
    return [ReminderOut(id=r.id, content=r.content, done=bool(r.done), created_at=r.created_at) for r in reminders]


@app.post("/api/reminders", response_model=ReminderOut)
async def create_reminder(
    req: ReminderCreateRequest,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    reminder = Reminder(user_id=current_user.id, content=req.content)
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return ReminderOut(id=reminder.id, content=reminder.content, done=bool(reminder.done), created_at=reminder.created_at)


@app.patch("/api/reminders/{reminder_id}/toggle", response_model=ReminderOut)
async def toggle_reminder(
    reminder_id: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    reminder = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if reminder is None or reminder.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reminder not found")
    reminder.done = 0 if reminder.done else 1
    db.commit()
    db.refresh(reminder)
    return ReminderOut(id=reminder.id, content=reminder.content, done=bool(reminder.done), created_at=reminder.created_at)


@app.delete("/api/reminders/{reminder_id}")
async def delete_reminder(
    reminder_id: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    reminder = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if reminder is None or reminder.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reminder not found")
    db.delete(reminder)
    db.commit()
    return {"deleted": True}


# ---------- conversations & chat (require auth) ----------


@app.post("/api/conversations")
async def create_conversation(
    req: NewConversationRequest,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    try:
        get_role(req.role_id)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown role_id: {req.role_id}")

    conv = Conversation(user_id=current_user.id, role_id=req.role_id)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"conversation_id": conv.id, "role_id": conv.role_id}


@app.get("/api/conversations", response_model=list[ConversationSummaryOut])
async def list_conversations(current_user: User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )


@app.get("/api/conversations/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv is None or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


async def _extract_and_store_memories_task(user_id: int, conversation_id: str, message: str):
    """
    Runs detached from the request/response cycle (fire-and-forget via
    asyncio.create_task) so memory extraction — which can involve an extra
    LLM call — never adds latency to the visible chat reply. Opens its own
    DB session since the request-scoped one is gone by the time this runs.
    Best-effort: any failure here is swallowed rather than surfaced, since
    there's no request left to report it to.
    """
    db = SessionLocal()
    try:
        extraction = await memory_module.extract_memories(message)
        for fact in extraction["facts"]:
            fact_embed = await rag.get_embedding(fact)
            nearest = await rag.find_relevant(
                db,
                Memory,
                Memory.user_id == user_id,
                fact,
                top_k=1,
                min_similarity=0.0,
                query_vector=fact_embed["vector"],
            )
            if nearest and nearest[0][0] > 0.92:
                continue  # too similar to something already remembered
            db.add(
                Memory(
                    user_id=user_id,
                    content=fact,
                    embedding=fact_embed["vector"],
                    source_conversation_id=conversation_id,
                )
            )
        db.commit()
    except Exception:
        pass
    finally:
        db.close()


def _run_tool(detection: dict, user_id: int, db: Session) -> dict:
    """
    Executes a detected, non-network tool synchronously (calculator,
    reminders — both instant, no reason to be async). Returns
    {"tool": str, "result": str} or None if the tool didn't actually fire
    (e.g. calculator pattern matched but the expression didn't parse).
    """
    kind = detection["tool"]

    if kind == "calculator":
        try:
            value = tools.safe_calculate(detection["expression"])
            return {"tool": "calculator", "result": f"{detection['expression'].strip()} = {value}"}
        except Exception:
            return None

    if kind == "reminder_add":
        reminder = Reminder(user_id=user_id, content=detection["content"])
        db.add(reminder)
        db.commit()
        return {"tool": "reminder_add", "result": f'Added reminder: "{detection["content"]}"'}

    if kind == "reminder_list":
        reminders = (
            db.query(Reminder)
            .filter(Reminder.user_id == user_id, Reminder.done == 0)
            .order_by(Reminder.created_at.desc())
            .all()
        )
        if not reminders:
            return {"tool": "reminder_list", "result": "No reminders are currently saved."}
        listing = "; ".join(r.content for r in reminders)
        return {"tool": "reminder_list", "result": f"Current reminders: {listing}"}

    return None


async def _stream_chat_response(conversation_id: str, role: dict, history: list, user_message: str, user_id: int):
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    db = SessionLocal()
    try:
        # Tool detection: calculator/reminders run instantly and locally;
        # weather needs a network call. If a tool fires, its result grounds
        # the reply directly and we skip document/memory retrieval — the
        # tool result already answers the message, so there's no reason to
        # spend time (and, with a remote embedding provider, an extra embedding call) on
        # retrieval that won't be used.
        detection = tools.detect_tool(user_message)
        tool_result = None
        if detection["tool"] in ("calculator", "reminder_add", "reminder_list"):
            tool_result = _run_tool(detection, user_id, db)
        elif detection["tool"] == "weather":
            try:
                weather = await tools.fetch_weather(detection["location"])
                tool_result = {
                    "tool": "weather",
                    "result": (
                        f"Weather in {weather['place']}: {weather['temperature_c']}°C, "
                        f"humidity {weather['humidity_pct']}%, wind {weather['wind_kmh']} km/h."
                    ),
                }
            except Exception:
                tool_result = None  # unknown place / network issue — fall through to a normal reply

        retrieved = []
        memory_context = []

        if tool_result is None:
            query_vector = (await rag.get_embedding(user_message))["vector"]

            doc_matches = await rag.find_relevant(
                db, DocumentChunk, DocumentChunk.user_id == user_id, user_message, query_vector=query_vector
            )
            retrieved = [
                {"content": c.content, "document_id": c.document_id, "chunk_index": c.chunk_index}
                for _, c in doc_matches
            ]

            memory_matches = await rag.find_relevant(
                db, Memory, Memory.user_id == user_id, user_message, query_vector=query_vector
            )
            memory_context = [{"content": m.content} for _, m in memory_matches]

        accumulated = ""
        source = "openrouter"
        try:
            async for piece in llm.stream_openrouter_reply(
                role["system_prompt"], history, user_message, retrieved, memory_context, tool_result
            ):
                accumulated += piece
                yield sse({"type": "token", "text": piece})
        except Exception as provider_error:
            # Log the technical detail server-side; return a safe, actionable message.
            import logging
            logging.getLogger("voxia.llm").exception("OpenRouter generation failed")
            yield sse({"type": "error", "detail": "AI provider request failed. Check the OpenRouter key, model availability, rate limits, and backend logs."})
            return

        if not accumulated.strip():
            yield sse({"type": "error", "detail": "The AI provider returned an empty response. Please try again."})
            return

        db.add(Message(conversation_id=conversation_id, sender="assistant", content=accumulated))
        db.commit()

        doc_sources = []
        if retrieved:
            doc_ids = {c["document_id"] for c in retrieved}
            docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
            doc_sources = [d.filename for d in docs]

        yield sse(
            {
                "type": "done",
                "source": source,
                "conversation_id": conversation_id,
                "rag_sources": doc_sources,
                "memories_used": [m["content"] for m in memory_context],
                "tool_used": tool_result["tool"] if tool_result else None,
            }
        )

        # Fire-and-forget: never blocks the reply the user is reading.
        asyncio.create_task(_extract_and_store_memories_task(user_id, conversation_id, user_message))
    except Exception as e:
        import logging
        logging.getLogger("voxia.chat").exception("Chat stream failed")
        yield sse({"type": "error", "detail": "The request could not be completed. Please retry; details are recorded in the server logs."})
    finally:
        db.close()


@app.post("/api/chat")
async def chat(
    req: ChatRequest,
    current_user: User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    # Fail fast when the provider is not configured. This prevents local tools,
    # retrieval, or other preprocessing from making the chat appear operational.
    if not llm.OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenRouter is not configured. Add OPENROUTER_API_KEY to the backend environment and restart the server.",
        )

    conv = db.query(Conversation).filter(Conversation.id == req.conversation_id).first()
    if conv is None or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    role = get_role(conv.role_id)
    history = [{"sender": m.sender, "content": m.content} for m in conv.messages]

    db.add(Message(conversation_id=conv.id, sender="user", content=req.message))
    db.commit()

    return StreamingResponse(
        _stream_chat_response(
            conversation_id=conv.id,
            role=role,
            history=history,
            user_message=req.message,
            user_id=current_user.id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
