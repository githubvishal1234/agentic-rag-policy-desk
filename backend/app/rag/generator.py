import os
from functools import cache

import anthropic

from app.rag import openai_compat

# Claude model; LLM_MODEL overrides it (and names the model for other providers).
MODEL = "claude-opus-5-5"


def provider() -> str:
    """"anthropic" (default) or "openai" for any OpenAI-compatible API."""

    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def claude_model() -> str:
    return os.environ.get("LLM_MODEL") or MODEL

SYSTEM_PROMPT = (
    "You answer employee questions using only the provided company "
    "documents. If the documents do not contain the answer, say that you "
    "could not find it in the documents. Keep answers concise."
)


@cache
def get_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) on first use.
    return anthropic.Anthropic()


def has_credentials() -> bool:
    """True when the selected provider has what it needs to answer."""

    if provider() == "openai":
        return openai_compat.is_configured()

    # Claude: an API key, auth token or login profile.
    client = get_client()
    return bool(client.api_key or client.auth_token or client.credentials)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Answer a question from retrieved chunks, citing the chunks used."""

    if provider() == "openai":
        return openai_compat.answer_from_chunks(SYSTEM_PROMPT, question, chunks)

    documents = [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": chunk["text"],
            },
            "title": chunk["metadata"]["source"],
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]

    response = get_client().beta.messages.create(
        model=claude_model(),
        max_tokens=16000,
        output_config={"effort": "medium"},
        # Retry on another model if a safety classifier declines.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [*documents, {"type": "text", "text": question}],
            }
        ],
    )

    if response.stop_reason == "refusal":
        return {
            "answer": "I can't help with that request.",
            "citations": [],
        }

    answer = ""
    citations = []

    for block in response.content:
        if block.type != "text":
            continue

        answer += block.text

        for citation in block.citations or []:
            citations.append(
                {
                    "cited_text": citation.cited_text,
                    **chunks[citation.document_index]["metadata"],
                }
            )

    return {"answer": answer, "citations": citations}
