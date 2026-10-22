import json
from threading import Lock
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient
from anthropic import DefaultHttpxClient

from app.agents import rag_agent
from app.services.ingestion import ingest_file
from app.services.vector_store import FaissVectorStore


def reply(*content, stop_reason="end_turn"):
    """A Messages API response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": list(content),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def tool_use(tool_id, name, tool_input):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def as_event_stream(message):
    """The Messages API streaming events that build up `message`."""
    events = [{"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}]

    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            start = {"type": "text", "text": ""}
            if "citations" in block:
                start["citations"] = []
            # Two text deltas, to check the endpoint forwards each piece.
            half = len(block["text"]) // 2
            deltas = [
                {"type": "text_delta", "text": block["text"][:half]},
                {"type": "text_delta", "text": block["text"][half:]},
            ] + [{"type": "citations_delta", "citation": c} for c in block.get("citations", [])]
        elif block["type"] == "tool_use":
            start = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        else:  # thinking
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]

        events.append({"type": "content_block_start", "index": index, "content_block": start})
        events += [{"type": "content_block_delta", "index": index, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": index})

    events += [
        {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@pytest.fixture
def claude(monkeypatch):
    """Scripted Claude: queue replies in .responses; requests land in .requests."""
    script = SimpleNamespace(responses=[], requests=[])

    def handler(request):
        body = json.loads(request.content)
        script.requests.append(body)
        message = script.responses.pop(0)

        if body.get("stream"):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=as_event_stream(message).encode(),
            )

        return httpx2.Response(200, json=message)

    client = anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    monkeypatch.setattr(rag_agent, "get_client", lambda: client)
    return script


@pytest.fixture
def agent(docs_dir, embedding_service):
    store = FaissVectorStore(embedding_service)
    store.add_documents(
        ingest_file(str(docs_dir / "company_it_policy.txt"), chunk_size=200, chunk_overlap=40)
    )
    return rag_agent.build_agent(store, Lock())


def ask(agent, message, thread_id="thread-1"):
    state = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": thread_id}},
    )
    return rag_agent.summarize_turn(state["messages"])


def test_tool_loop_with_citations(agent, claude):
    claude.responses = [
        reply(
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            tool_use("toolu_1", "search_documents", {"query": "lost laptop"}),
            tool_use("toolu_2", "list_documents", {}),
            stop_reason="tool_use",
        ),
        reply(
            {
                "type": "text",
                "text": "Report it to IT immediately.",
                "citations": [
                    {
                        "type": "search_result_location",
                        "cited_text": "Employees must report lost or stolen devices",
                        "source": "company_it_policy.txt",
                        "title": "company_it_policy.txt (chunk 2)",
                        "search_result_index": 0,
                        "start_block_index": 0,
                        "end_block_index": 1,
                    }
                ],
            }
        ),
    ]

    result = ask(agent, "What do I do if I lose my laptop?")

    assert result["answer"] == "Report it to IT immediately."
    assert result["segments"] == [
        {"text": "Report it to IT immediately.", "citations": [0]}
    ]
    assert result["citations"][0]["source"] == "company_it_policy.txt"
    assert result["tool_calls"] == [
        {"name": "search_documents", "input": {"query": "lost laptop"}},
        {"name": "list_documents", "input": {}},
    ]

    # Second request replays the assistant turn unchanged (thinking
    # signature included) and returns both tool results in one message.
    _, assistant, tool_results = claude.requests[1]["messages"]
    assert assistant["content"][0]["signature"] == "sig-1"

    search, listing = tool_results["content"]
    assert search["tool_use_id"] == "toolu_1"
    assert search["content"][0]["type"] == "search_result"
    assert search["content"][0]["citations"] == {"enabled": True}
    assert any(
        "lost or stolen" in part["text"]
        for block in search["content"]
        for part in block["content"]
    )
    assert listing["content"] == [{"type": "text", "text": "company_it_policy.txt"}]

    assert claude.requests[0]["fallbacks"] == "default"
    assert all(tool["strict"] for tool in claude.requests[0]["tools"])


def test_conversation_memory(agent, claude):
    claude.responses = [
        reply({"type": "text", "text": "Hello!"}),
        reply({"type": "text", "text": "You said hi."}),
    ]

    ask(agent, "Hi")
    result = ask(agent, "What did I just say?")

    assert result["answer"] == "You said hi."
    assert result["tool_calls"] == []
    assert [message["role"] for message in claude.requests[1]["messages"]] == [
        "user", "assistant", "user",
    ]

    # A different thread starts fresh.
    claude.responses = [reply({"type": "text", "text": "New thread."})]
    ask(agent, "Hi again", thread_id="thread-2")
    assert len(claude.requests[2]["messages"]) == 1


def test_tool_error_is_reported_to_claude(agent, claude):
    claude.responses = [
        reply(tool_use("toolu_1", "search_documents", {"query": "   "}), stop_reason="tool_use"),
        reply({"type": "text", "text": "Please rephrase."}),
    ]

    ask(agent, "?")

    result = claude.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "empty" in result["content"]


def test_refusal_keeps_thread_usable(agent, claude):
    claude.responses = [
        reply(stop_reason="refusal"),
        reply({"type": "text", "text": "Sure."}),
    ]

    assert ask(agent, "Something declined")["answer"] == "I can't help with that request."
    assert ask(agent, "Something else")["answer"] == "Sure."


def test_chat_endpoint(client, claude, docs_dir):
    with (docs_dir / "company_it_policy.txt").open("rb") as file:
        client.post("/ingest", files={"file": ("company_it_policy.txt", file)})

    claude.responses = [
        reply(tool_use("toolu_1", "search_documents", {"query": "VPN"}), stop_reason="tool_use"),
        reply({"type": "text", "text": "Use the approved VPN."}),
        reply({"type": "text", "text": "Yes."}),
    ]

    first = client.post("/chat", json={"message": "How do I work remotely?"}).json()
    assert first["answer"] == "Use the approved VPN."
    assert first["tool_calls"] == [{"name": "search_documents", "input": {"query": "VPN"}}]

    second = client.post(
        "/chat",
        json={"message": "Is that required?", "thread_id": first["thread_id"]},
    ).json()
    assert second["thread_id"] == first["thread_id"]
    assert len(claude.requests[-1]["messages"]) == 5

    assert client.post("/chat", json={"message": "  "}).status_code == 422


def test_conversation_survives_restart(client, claude, docs_dir):
    with (docs_dir / "company_it_policy.txt").open("rb") as file:
        client.post("/ingest", files={"file": ("company_it_policy.txt", file)})

    claude.responses = [
        reply(tool_use("toolu_1", "search_documents", {"query": "VPN"}), stop_reason="tool_use"),
        reply({"type": "text", "text": "Use the approved VPN."}),
    ]
    thread_id = client.post("/chat", json={"message": "How do I work remotely?"}).json()["thread_id"]

    # A new app instance reads the same SQLite file, like a server restart.
    from app import main
    with TestClient(main.app) as restarted:
        history = restarted.get(f"/chat/{thread_id}").json()

        claude.responses = [reply({"type": "text", "text": "Yes, always."})]
        restarted.post("/chat", json={"message": "Even at home?", "thread_id": thread_id})

        assert restarted.get("/chat/unknown-thread").status_code == 404

    assert [turn["question"] for turn in history["turns"]] == ["How do I work remotely?"]
    assert history["turns"][0]["answer"] == "Use the approved VPN."
    assert history["turns"][0]["tool_calls"] == [{"name": "search_documents", "input": {"query": "VPN"}}]
    # The follow-up was sent with the earlier turn as context.
    assert len(claude.requests[-1]["messages"]) == 5


def test_conversation_turns_skips_unanswered_turns():
    messages = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": [{"type": "text", "text": "One."}]},
        {"role": "user", "content": "Second?"},  # the LLM call failed here
        {"role": "user", "content": "Third?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Three."}]},
    ]

    turns = rag_agent.conversation_turns(messages)

    assert [(turn["question"], turn["answer"]) for turn in turns] == [
        ("First?", "One."),
        ("Third?", "Three."),
    ]


def stream_events(client, body):
    response = client.post("/chat/stream", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_chat_stream(client, claude, docs_dir):
    with (docs_dir / "company_it_policy.txt").open("rb") as file:
        client.post("/ingest", files={"file": ("company_it_policy.txt", file)})

    claude.responses = [
        reply(
            {"type": "thinking", "thinking": "", "signature": "sig-1"},
            tool_use("toolu_1", "search_documents", {"query": "VPN"}),
            stop_reason="tool_use",
        ),
        reply({
            "type": "text",
            "text": "Use the approved VPN.",
            "citations": [{
                "type": "search_result_location",
                "cited_text": "Employees accessing company systems remotely must use the approved VPN.",
                "source": "company_it_policy.txt",
                "title": "company_it_policy.txt (chunk 1)",
                "search_result_index": 0,
                "start_block_index": 0,
                "end_block_index": 1,
            }],
        }),
    ]

    events = stream_events(client, {"message": "How do I work remotely?"})

    assert [e["type"] for e in events] == ["tool", "text", "text", "done"]
    assert events[0] == {"type": "tool", "name": "search_documents", "input": {"query": "VPN"}}
    assert "".join(e["delta"] for e in events if e["type"] == "text") == "Use the approved VPN."

    done = events[-1]
    assert done["answer"] == "Use the approved VPN."
    assert done["citations"][0]["source"] == "company_it_policy.txt"
    assert done["segments"] == [{"text": "Use the approved VPN.", "citations": [0]}]
    assert all(request["stream"] for request in claude.requests)

    # The streamed turn was saved like any other: the thread continues.
    assert [t["question"] for t in client.get(f"/chat/{done['thread_id']}").json()["turns"]] == [
        "How do I work remotely?"
    ]
    # The thinking signature survived stream accumulation for replay.
    assert claude.requests[1]["messages"][1]["content"][0]["signature"] == "sig-1"


def test_chat_stream_reports_llm_errors(client, monkeypatch):
    def failing_client():
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))

    monkeypatch.setattr(rag_agent, "get_client", failing_client)

    events = stream_events(client, {"message": "Hi"})

    assert events[-1]["type"] == "error"
    assert "LLM request failed" in events[-1]["detail"]



def test_endless_tool_calls_end_the_turn_cleanly(agent, claude):
    # Claude keeps searching forever: one more reply than the cap allows.
    claude.responses = [
        reply(tool_use(f"toolu_{n}", "search_documents", {"query": f"try {n}"}), stop_reason="tool_use")
        for n in range(rag_agent.MAX_TOOL_ROUNDS + 1)
    ]

    result = ask(agent, "Impossible question?")

    assert result["answer"] == rag_agent.GAVE_UP_ANSWER
    assert len(result["tool_calls"]) == rag_agent.MAX_TOOL_ROUNDS + 1
    assert claude.responses == []

    # The thread still works: every tool call in the history has a result.
    claude.responses = [reply({"type": "text", "text": "Fine now."})]
    assert ask(agent, "Something simple?")["answer"] == "Fine now."

    history = claude.requests[-1]["messages"]
    tool_ids = {b["id"] for m in history if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"}
    result_ids = {b["tool_use_id"] for m in history if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_result"}
    assert tool_ids == result_ids
    assert [m["role"] for m in history[-3:]] == ["user", "assistant", "user"]
