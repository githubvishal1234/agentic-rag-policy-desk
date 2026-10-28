"""The OpenAI-compatible provider (OpenAI, Gemini, Groq, Ollama, ...)."""

import json
from threading import Lock
from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai import DefaultHttpx2Client

from app.agents import rag_agent
from app.rag import generator, openai_compat
from app.services.ingestion import ingest_file
from app.services.vector_store import FaissVectorStore

SOURCES = [
    {"cited_text": "Leave text", "source": "leave.txt", "title": "leave.txt (chunk 0)"},
    {"cited_text": "VPN text", "source": "it.txt", "title": "it.txt (chunk 2)"},
]


def test_split_citations():
    blocks = openai_compat.split_citations(
        "You get 22 days [1]. Use the VPN [2, 1][9]. Ask HR.", SOURCES
    )

    assert [(b["text"], [c["source"] for c in b["citations"]]) for b in blocks] == [
        ("You get 22 days", ["leave.txt"]),
        (". Use the VPN", ["it.txt", "leave.txt"]),  # [9] doesn't exist: dropped
        (". Ask HR.", []),
    ]


def test_split_citations_without_markers():
    assert openai_compat.split_citations("No sources.", SOURCES) == [
        {"type": "text", "text": "No sources.", "citations": []}
    ]


def completion(content=None, tool_calls=None):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        }],
    }


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture
def chat_api(monkeypatch):
    """Scripted OpenAI-compatible API: queue .responses, read .requests."""
    script = SimpleNamespace(responses=[], requests=[])

    def handler(request):
        script.requests.append(json.loads(request.content))
        return httpx2.Response(200, json=script.responses.pop(0))

    client = openai.OpenAI(
        api_key="test",
        base_url="http://llm.test/v1",
        http_client=DefaultHttpx2Client(transport=httpx2.MockTransport(handler)),
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(openai_compat, "get_client", lambda: client)
    return script


def test_has_credentials(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    for name in ["LLM_MODEL", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)

    assert not generator.has_credentials()

    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    assert not generator.has_credentials()

    # Ollama: a local URL is enough, no key.
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert generator.has_credentials()


def test_query_answer_with_citations(chat_api):
    chunks = [
        {"text": "Employees get 22 days.", "metadata": {"source": "leave.txt", "chunk_id": 0}, "score": 0.9},
        {"text": "Use the VPN.", "metadata": {"source": "it.txt", "chunk_id": 3}, "score": 0.5},
    ]
    chat_api.responses = [completion("You get 22 days [1].")]

    result = generator.generate_answer("How much leave?", chunks)

    assert result == {
        "answer": "You get 22 days.",
        "citations": [{"cited_text": "Employees get 22 days.", "source": "leave.txt", "chunk_id": 0}],
    }
    request = chat_api.requests[0]
    assert request["model"] == "test-model"
    assert "[1] leave.txt\nEmployees get 22 days." in request["messages"][1]["content"]
    assert "cite it with its number" in request["messages"][0]["content"]


@pytest.fixture
def agent(docs_dir, embedding_service):
    store = FaissVectorStore(embedding_service)
    store.add_documents(ingest_file(str(docs_dir / "company_it_policy.txt"), chunk_size=200, chunk_overlap=40))
    return rag_agent.build_agent(store, Lock())


def ask(agent, message):
    state = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": "t1"}},
    )
    return rag_agent.summarize_turn(state["messages"])


def test_agent_tool_loop_and_citations(agent, chat_api):
    chat_api.responses = [
        completion(tool_calls=[tool_call("call_1", "search_documents", '{"query": "lost laptop"}')]),
        completion("Report it to IT within 2 hours [1]."),
        completion("You're welcome."),
    ]

    result = ask(agent, "I lost my laptop.")

    assert result["answer"] == "Report it to IT within 2 hours."
    assert result["tool_calls"] == [{"name": "search_documents", "input": {"query": "lost laptop"}}]
    assert result["citations"][0]["source"] == "company_it_policy.txt"
    assert result["segments"][0]["citations"] == [0]

    # The second request carries the tool call and its numbered results.
    messages = chat_api.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "search_documents"
    assert messages[3]["tool_call_id"] == "call_1"
    assert messages[3]["content"].startswith("[1] company_it_policy.txt")
    assert chat_api.requests[0]["tools"][0]["function"]["name"] == "search_documents"

    # History replays cleanly on the next turn.
    assert ask(agent, "Thanks!")["answer"] == "You're welcome."
    assert [m["role"] for m in chat_api.requests[2]["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "user",
    ]


def test_agent_survives_bad_tool_arguments(agent, chat_api):
    chat_api.responses = [
        completion(tool_calls=[tool_call("call_1", "search_documents", "{not json")]),
        completion("Sorry, please rephrase."),
    ]

    assert ask(agent, "?")["answer"] == "Sorry, please rephrase."
    tool_message = chat_api.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "Invalid tool call" in tool_message["content"]
