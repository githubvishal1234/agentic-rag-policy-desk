import operator
from threading import Lock
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.rag.generator import MODEL, get_client
from app.services.vector_store import FaissVectorStore

SYSTEM_PROMPT = (
    "You are a company knowledge assistant. Answer employee questions using "
    "only what the search_documents tool returns. Search before answering "
    "any question about company policy; for questions with several parts, "
    "search for each part. If the documents do not contain the answer, say "
    "that you could not find it in the documents. Keep answers concise."
)

TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the company knowledge base (policies, handbooks) and "
            "return the most relevant passages. Use specific queries; call "
            "it again with different wording if the results miss."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "list_documents",
        "description": "List the file names in the knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class AgentState(TypedDict):
    # Anthropic Messages API messages, only ever appended to: rewriting
    # earlier turns would invalidate the model's thinking blocks.
    messages: Annotated[list[dict], operator.add]


def chunk_title(metadata: dict) -> str:
    title = metadata["source"]

    if "page" in metadata:
        title += f", page {metadata['page']}"

    return f"{title} (chunk {metadata['chunk_id']})"


def build_agent(store: FaissVectorStore, store_lock: Lock, checkpointer=None):
    """Compile the agent graph: model -> tools -> model ... -> END.

    `checkpointer` stores conversations by thread_id; the default keeps them
    in memory only. The API passes a SqliteSaver so they survive restarts.
    """

    def search_documents(query: str) -> list[dict]:
        with store_lock:
            chunks = store.hybrid_search(query, k=5)

        if not chunks:
            return [{"type": "text", "text": "No matching documents found."}]

        return [
            {
                "type": "search_result",
                "source": chunk["metadata"]["source"],
                "title": chunk_title(chunk["metadata"]),
                # One block per paragraph: a citation covers whole blocks,
                # so this lets answers quote just the paragraph they used.
                "content": [
                    {"type": "text", "text": paragraph.strip()}
                    for paragraph in chunk["text"].split("\n\n")
                    if paragraph.strip()
                ],
                "citations": {"enabled": True},
            }
            for chunk in chunks
        ]

    def list_documents() -> list[dict]:
        with store_lock:
            sources = sorted(
                {document.metadata["source"] for document in store.documents}
            )

        text = "\n".join(sources) or "The knowledge base is empty."
        return [{"type": "text", "text": text}]

    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        params = dict(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            # Retry on another model if a safety classifier declines.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=state["messages"],
        )

        if config["configurable"].get("stream_text"):
            # Forward answer text as it is generated; graph.stream() callers
            # receive it in stream_mode "custom".
            write = get_stream_writer()

            with get_client().beta.messages.stream(**params) as stream:
                for event in stream:
                    if event.type == "text":
                        write({"type": "text", "delta": event.text})

                response = stream.get_final_message()
        else:
            response = get_client().beta.messages.create(**params)

        if response.stop_reason == "refusal":
            content = [
                {"type": "text", "text": "I can't help with that request."}
            ]
        else:
            # Keep every block (thinking included) so later turns can
            # replay them unchanged; dicts keep the checkpoint serializable.
            content = [
                block.model_dump(mode="json", exclude_none=True)
                for block in response.content
            ]

        return {"messages": [{"role": "assistant", "content": content}]}

    def run_tools(state: AgentState) -> dict:
        results = []

        for block in state["messages"][-1]["content"]:
            if block["type"] != "tool_use":
                continue

            result = {"type": "tool_result", "tool_use_id": block["id"]}

            try:
                if block["name"] == "search_documents":
                    result["content"] = search_documents(
                        block["input"]["query"]
                    )
                elif block["name"] == "list_documents":
                    result["content"] = list_documents()
                else:
                    raise ValueError(f"Unknown tool: {block['name']}")
            except ValueError as error:
                result["content"] = str(error)
                result["is_error"] = True

            results.append(result)

        # All results go back in one message so Claude keeps calling
        # tools in parallel.
        return {"messages": [{"role": "user", "content": results}]}

    def has_tool_calls(state: AgentState) -> str:
        content = state["messages"][-1]["content"]

        if any(block["type"] == "tool_use" for block in content):
            return "tools"

        return END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", has_tool_calls, ["tools", END])
    graph.add_edge("tools", "model")

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def summarize_turn(messages: list[dict]) -> dict:
    """Extract the final answer, its citations, and this turn's tool calls."""

    # The turn starts at the latest user message with plain-text content;
    # tool results are user messages too, but their content is a list.
    start = max(
        index
        for index, message in enumerate(messages)
        if message["role"] == "user" and isinstance(message["content"], str)
    )

    tool_calls = [
        {"name": block["name"], "input": block["input"]}
        for message in messages[start:]
        if message["role"] == "assistant"
        for block in message["content"]
        if block["type"] == "tool_use"
    ]

    answer = ""
    citations = []
    # The answer split where Claude attached citations; each segment lists
    # indexes into `citations`, so a UI can place markers inline.
    segments = []

    for block in messages[-1]["content"]:
        if block["type"] != "text":
            continue

        answer += block["text"]
        references = []

        for citation in block.get("citations", []):
            entry = {
                "cited_text": citation["cited_text"],
                "source": citation["source"],
                "title": citation.get("title"),
            }

            if entry not in citations:
                citations.append(entry)

            references.append(citations.index(entry))

        segments.append({"text": block["text"], "citations": references})

    return {
        "answer": answer,
        "segments": segments,
        "citations": citations,
        "tool_calls": tool_calls,
    }


def conversation_turns(messages: list[dict]) -> list[dict]:
    """Summarize every completed turn: the question plus summarize_turn()."""

    turns = []

    for index, message in enumerate(messages):
        if message["role"] != "user" or not isinstance(message["content"], str):
            continue

        # The turn runs until the next plain-text user message.
        end = next(
            (
                later
                for later in range(index + 1, len(messages))
                if messages[later]["role"] == "user"
                and isinstance(messages[later]["content"], str)
            ),
            len(messages),
        )

        # Skip turns that never got a final answer (e.g. the LLM call failed).
        if messages[end - 1]["role"] != "assistant":
            continue

        turns.append(
            {"question": message["content"], **summarize_turn(messages[:end])}
        )

    return turns

