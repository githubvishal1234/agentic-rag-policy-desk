from functools import cache

import anthropic

MODEL = "claude-opus-5-5"

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
    """True when the client found an API key, auth token or login profile."""

    client = get_client()
    return bool(client.api_key or client.auth_token or client.credentials)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Answer a question from retrieved chunks, citing the chunks used."""

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
        model=MODEL,
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
