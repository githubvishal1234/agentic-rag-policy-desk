# Policy Desk: Agentic RAG over company documents

[![CI](https://github.com/githubvishal1234/agentic-rag-policy-desk/actions/workflows/ci.yml/badge.svg)](https://github.com/githubvishal1234/agentic-rag-policy-desk/actions/workflows/ci.yml)

Employees ask questions about company policy; an AI agent searches the
documents and answers, citing the exact passage behind each claim.

Built with Python, FastAPI, LangGraph, FAISS, hybrid retrieval, tool
calling and React. Works with Claude or any OpenAI-compatible model
(OpenAI, Gemini, Groq, or Ollama running locally).

**→ Full setup and usage guide: [RUNNING.md](RUNNING.md)**

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
npm install --prefix frontend
copy .env.example .env          # then add your API key, see RUNNING.md section 3
npm run build --prefix frontend
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
```

Open http://localhost:8000, add documents (samples in `data/sample` and
`data/eval_corpus`), and ask a question. Or with Docker:

```powershell
docker build -t policy-desk .
docker run -p 8000:8000 --env-file .env -v policy-desk-data:/app/data policy-desk
```

## Features

- **Document ingestion**: `.txt`, `.pdf` and `.docx`, chunked and indexed
  locally (sentence-transformers embeddings, FAISS). Add and remove files
  from the web app.
- **Hybrid retrieval**: vector search plus BM25 keyword search, merged with
  Reciprocal Rank Fusion, so exact terms (form codes, acronyms) and
  paraphrases both match.
- **Agent with tool calling**: a LangGraph loop where the model decides when
  and how often to search; multi-part questions get one search per part.
  Tool rounds are capped so a runaway loop ends cleanly.
- **Citations**: every answer links each claim to the passage it came from,
  shown as margin notes beside the answer.
- **Streaming**: searches and answer text appear as the agent works.
- **Conversations**: follow-up questions keep context; conversations are
  stored in SQLite and survive restarts.
- **Sign-in**: optional shared passwords with employee and admin roles.
- **Pluggable LLM**: Claude by default (native citations), or any
  OpenAI-compatible API through three settings.

## How it works

```
upload ─► load (txt/pdf/docx) ─► chunk ─► embed ─► FAISS + BM25 index
                                                         │
question ─► LangGraph agent ──► search_documents tool ───┘
               │    ▲                 (hybrid search, RRF)
               ▼    │
            LLM (Claude / OpenAI-compatible) ─► answer + citations ─► Policy Desk
```

| Path | Contents |
|---|---|
| `backend/app/services/` | Loading, chunking, embeddings, vector store, BM25 |
| `backend/app/agents/rag_agent.py` | The LangGraph agent and its tools |
| `backend/app/rag/` | One-shot answers (`/query`) and the LLM providers |
| `backend/app/main.py` | FastAPI app and endpoints |
| `backend/app/core/auth.py` | Sign-in |
| `frontend/` | Policy Desk (React + Vite) |
| `evaluation/` | Eval cases and runners |
| `data/eval_corpus/` | 11 fictional policy documents used by the evals |

## Results

On 26 answerable questions over the eval corpus (lookups, paraphrases,
exact terms such as form codes, and multi-part questions), the passage
holding the answer is in the top 4 search results for:

| Retrieval | Found |
|---|---|
| Vector search only | 24 / 26 |
| Hybrid (vector + BM25) | 24 / 26 |
| Hybrid + stopword filtering | **26 / 26** |

The answer-quality eval (facts correct, right source cited, no invented
answers to unanswerable questions) is built and ready to run with an API
key: see [RUNNING.md, section 9](RUNNING.md#9-evaluations).

## Tests

58 backend tests (fake LLM responses, no key needed) plus a frontend build
and a Docker build-and-run check run on every push:

```powershell
cd backend
..\.venv\Scripts\python -m pytest
```
