import sqlite3
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, Field

from app.agents.rag_agent import build_agent, conversation_turns, summarize_turn
from app.rag.generator import generate_answer, has_credentials
from app.services.embedding_service import EmbeddingService
from app.services.ingestion import ingest_file
from app.services.vector_store import FaissVectorStore

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
SUPPORTED_TYPES = {".txt", ".pdf", ".docx"}

# ponytail: one lock for every store read/write; ingesting a large file
# blocks queries until it finishes. Use a read/write lock if that hurts.
store_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = FaissVectorStore(EmbeddingService())

    if (DATA_DIR / "vectorstore" / "index.faiss").exists():
        store.load(str(DATA_DIR / "vectorstore"))

    # Conversations persist in SQLite so threads survive restarts. FastAPI
    # runs handlers in worker threads; SqliteSaver locks the connection.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        DATA_DIR / "conversations.sqlite", check_same_thread=False
    )

    app.state.store = store
    app.state.agent = build_agent(store, store_lock, SqliteSaver(connection))
    yield
    connection.close()


app = FastAPI(
    title="Enterprise Agentic RAG Platform",
    version="0.1.0",
    description="Enterprise AI knowledge and data assistant",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1)
    thread_id: str | None = None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-agentic-rag",
        "llm_configured": has_credentials(),
    }


def require_credentials() -> None:
    if not has_credentials():
        raise HTTPException(
            503,
            "The server has no Anthropic API key. Set ANTHROPIC_API_KEY "
            "and restart the server.",
        )


@app.get("/documents")
def documents():
    with store_lock:
        counts = Counter(
            document.metadata["source"]
            for document in app.state.store.documents
        )

    return [
        {"source": source, "chunks": chunks}
        for source, chunks in sorted(counts.items())
    ]


@app.post("/ingest")
def ingest(file: UploadFile):
    # Keep only the base name so uploads cannot escape the documents folder.
    file_name = Path(file.filename or "").name

    if Path(file_name).suffix.lower() not in SUPPORTED_TYPES:
        raise HTTPException(400, "Supported types: .txt, .pdf, .docx")

    store = app.state.store

    with store_lock:
        if any(
            document.metadata["source"] == file_name
            for document in store.documents
        ):
            raise HTTPException(409, f"{file_name} is already ingested")

        path = DATA_DIR / "documents" / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.file.read())

        try:
            chunks = ingest_file(str(path))
        except Exception as error:
            path.unlink()
            raise HTTPException(422, f"Could not read {file_name}: {error}")

        store.add_documents(chunks)
        store.save(str(DATA_DIR / "vectorstore"))

    return {"source": file_name, "chunks": len(chunks)}


@app.post("/query")
def query(request: QueryRequest):
    with store_lock:
        chunks = app.state.store.hybrid_search(
            request.question,
            k=request.k,
        )

    if not chunks:
        return {
            "answer": "No documents have been ingested yet.",
            "citations": [],
            "sources": [],
        }

    require_credentials()

    try:
        result = generate_answer(request.question, chunks)
    except anthropic.APIError as error:
        raise HTTPException(502, f"LLM request failed: {error}")

    return {
        **result,
        "sources": [
            {**chunk["metadata"], "score": chunk["score"]}
            for chunk in chunks
        ],
    }


@app.post("/chat")
def chat(request: ChatRequest):
    require_credentials()
    thread_id = request.thread_id or str(uuid4())

    try:
        state = app.state.agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]},
            {"configurable": {"thread_id": thread_id}},
        )
    except anthropic.APIError as error:
        raise HTTPException(502, f"LLM request failed: {error}")
    except GraphRecursionError:
        raise HTTPException(500, "The agent took too many steps")

    return {"thread_id": thread_id, **summarize_turn(state["messages"])}



@app.get("/chat/{thread_id}")
def chat_history(thread_id: str):
    state = app.state.agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    if not messages:
        raise HTTPException(404, "Conversation not found")

    return {"thread_id": thread_id, "turns": conversation_turns(messages)}

# Serve the built frontend (npm run build) from the same server. Mounted
# last so the API routes above take precedence.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
