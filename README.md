# Enterprise Agentic RAG Platform

An enterprise AI knowledge and data assistant built with Python,
FastAPI, LangGraph, RAG, hybrid retrieval, tool calling, and React.

## Status

🚧 Under Development

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -r backend/requirements.txt
```

## Run the API

Answers are generated with Claude, so set an API key first
(get one at https://platform.claude.com):

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
cd backend
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs, or:

```bash
curl -F "file=@../data/sample/company_it_policy.txt" http://localhost:8000/ingest
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "What are the password rules?"}'
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "How many leave days do I get, and can I work from home?"}'
```

| Endpoint | Does |
|---|---|
| `POST /ingest` | Upload a .txt/.pdf/.docx file; it is chunked, embedded and saved to `data/vectorstore/` |
| `POST /query` | `{"question", "k"}` → hybrid search (vector + BM25 keyword, fused with RRF), then an answer with citations plus the retrieved sources |
| `POST /chat` | `{"message", "thread_id"?}` → LangGraph agent: Claude decides when and how often to search (`search_documents`, `list_documents` tools), remembers the conversation per `thread_id` (stored in `data/conversations.sqlite`, so it survives restarts), and answers with citations |
| `GET /chat/{thread_id}` | The conversation so far, turn by turn |
| `GET /health` | Health check |

## Frontend

Policy Desk, a React app in `frontend/`: upload documents, ask questions
(answered by the `/chat` agent), and check each claim against the quoted
passage in the margin.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies API calls to :8000
```

For a single server, run `npm run build`; FastAPI then serves the built app
from `frontend/dist` at http://localhost:8000.

## Tests

```bash
cd backend
pytest
```

Tests use the sample documents in `data/sample/`. Put your own documents in
`data/documents/` (not committed to git).

## Evaluation

`evaluation/cases.jsonl` holds 31 questions over a synthetic policy corpus in
`data/eval_corpus/` (see `evaluation/cases.md`). The retrieval eval checks
whether search puts each question's evidence passage in the top 4:

```bash
python evaluation/run_retrieval_eval.py --variant baseline --method vector
python evaluation/run_retrieval_eval.py --variant v2 --method hybrid
```

Results, per-case transcripts and `report.html` land in `.claude/hillclimb/retrieval/`.
Current scores: vector 24/26, hybrid with stopwords 26/26.

The answer eval runs the same cases through `/query` and `/chat` with real
Claude calls (needs `ANTHROPIC_API_KEY`; costs money, so pilot first):

```bash
python evaluation/run_answer_eval.py --variant baseline --app query --limit 5
python evaluation/run_answer_eval.py --variant v1 --app chat --limit 5
```

An answer passes when it contains every expected fact and cites every expected
source file; unanswerable questions pass when a Claude Haiku 5.5 judge confirms
the answer says it could not find the information. Output goes to
`.claude/hillclimb/answers/`; re-running resumes where it stopped.
