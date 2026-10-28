"""Any OpenAI-compatible chat API: OpenAI, Google Gemini, Groq, Ollama, ...

Selected with LLM_PROVIDER=openai, configured with:
    LLM_MODEL     model name, e.g. gpt-5-mini, gemini-2.5-flash, qwen2.5:7b
    LLM_BASE_URL  API URL; leave unset for OpenAI itself
    LLM_API_KEY   API key (or OPENAI_API_KEY); not needed for Ollama

These APIs have no native citations, so the model is asked to cite numbered
passages like [2]; the markers are parsed into the same citation format the
Claude path produces, so the agent state, API and frontend are unchanged.
"""

import json
import os
import re
from functools import cache

import openai

CITATION_INSTRUCTIONS = (
    " Passages are numbered like [3]. After each sentence that uses a "
    "passage, cite it with its number in square brackets, for example [3]. "
    "Cite only numbers you were given."
)

# [2] or [2, 5]
MARKER = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")


def model() -> str | None:
    return os.environ.get("LLM_MODEL")


def api_key() -> str | None:
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def is_configured() -> bool:
    # A local server such as Ollama needs a URL but no key.
    return bool(model() and (api_key() or os.environ.get("LLM_BASE_URL")))


@cache
def get_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=api_key() or "not-needed",
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )


def split_citations(text: str, sources: list[dict]) -> list[dict]:
    """Turn "A [1]. B [2, 3]." into text blocks carrying citations.

    `sources` are the numbered passages (1-based in the text), each a dict
    with cited_text, source and title. Unknown numbers are dropped.
    """

    blocks = []
    position = 0

    for match in MARKER.finditer(text):
        numbers = [int(n) for n in match.group(1).split(",")]
        citations = [
            sources[n - 1] for n in numbers if 1 <= n <= len(sources)
        ]
        segment = text[position:match.start()]
        position = match.end()

        # "A [1][2]": a marker straight after another cites the same text.
        if not segment and blocks:
            blocks[-1]["citations"] += citations
        else:
            blocks.append({"type": "text", "text": segment, "citations": citations})

    if position < len(text):
        blocks.append({"type": "text", "text": text[position:], "citations": []})

    return blocks or [{"type": "text", "text": "", "citations": []}]


def answer_from_chunks(system: str, question: str, chunks: list[dict]) -> dict:
    """The /query flow: one call over the retrieved chunks."""

    passages = "\n\n".join(
        f"[{number}] {chunk['metadata']['source']}\n{chunk['text']}"
        for number, chunk in enumerate(chunks, start=1)
    )
    response = get_client().chat.completions.create(
        model=model(),
        messages=[
            {"role": "system", "content": system + CITATION_INSTRUCTIONS},
            {
                "role": "user",
                "content": f"Passages:\n\n{passages}\n\nQuestion: {question}",
            },
        ],
    )

    sources = [
        {"cited_text": chunk["text"], **chunk["metadata"]} for chunk in chunks
    ]
    blocks = split_citations(response.choices[0].message.content or "", sources)

    citations = []
    for block in blocks:
        for citation in block["citations"]:
            if citation not in citations:
                citations.append(citation)

    return {
        "answer": "".join(block["text"] for block in blocks),
        "citations": citations,
    }


def render_result(block: dict, number: int | None) -> str:
    if block["type"] == "search_result":
        text = "\n".join(part["text"] for part in block["content"])
        return f"[{number}] {block['title']}\n{text}"

    return block.get("text", "")


def agent_step(system: str, tools: list[dict], messages: list[dict]) -> list[dict]:
    """One agent model call. Takes and returns the agent's message format
    (Anthropic-style content blocks) and converts at the API boundary."""

    chat = [{"role": "system", "content": system + CITATION_INSTRUCTIONS}]
    # Search results are numbered across the whole conversation, so a [n]
    # in the answer points at exactly one passage.
    sources = []

    for message in messages:
        if isinstance(message["content"], str):
            chat.append({"role": message["role"], "content": message["content"]})
            continue

        if message["role"] == "assistant":
            text = "".join(
                block["text"] for block in message["content"] if block["type"] == "text"
            )
            calls = [
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                }
                for block in message["content"]
                if block["type"] == "tool_use"
            ]
            entry = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            chat.append(entry)
            continue

        for block in message["content"]:
            if block["type"] != "tool_result":
                continue

            parts = block["content"]
            if isinstance(parts, str):
                parts = [{"type": "text", "text": parts}]

            rendered = []
            for part in parts:
                number = None
                if part["type"] == "search_result":
                    sources.append({
                        "cited_text": "\n".join(p["text"] for p in part["content"]),
                        "source": part["source"],
                        "title": part["title"],
                    })
                    number = len(sources)
                rendered.append(render_result(part, number))

            chat.append({
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": "\n\n".join(rendered),
            })

    response = get_client().chat.completions.create(
        model=model(),
        messages=chat,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ],
    )
    reply = response.choices[0].message

    content = split_citations(reply.content, sources) if reply.content else []

    for call in reply.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}  # run_tools reports the missing input as an error
        content.append({
            "type": "tool_use",
            "id": call.id,
            "name": call.function.name,
            "input": arguments,
        })

    # Never store an empty assistant turn: these APIs reject it on replay.
    return content or [{"type": "text", "text": "", "citations": []}]
