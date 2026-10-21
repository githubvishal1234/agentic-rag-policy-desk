from types import SimpleNamespace

from app.rag import generator

CHUNKS = [
    {
        "text": "Company laptops must use approved security software.",
        "metadata": {"source": "it_policy.txt", "chunk_id": 0},
        "score": 0.8,
    },
    {
        "text": "Employees get 20 days of annual leave.",
        "metadata": {"source": "leave_policy.pdf", "page": 1, "chunk_id": 0},
        "score": 0.2,
    },
]


def fake_client(response, calls):
    def create(**kwargs):
        calls.append(kwargs)
        return response

    return SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create))
    )


def test_answer_with_citations(monkeypatch):
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(
                type="text",
                text="Laptops need approved security software.",
                citations=[
                    SimpleNamespace(
                        cited_text=CHUNKS[0]["text"],
                        document_index=0,
                    )
                ],
            ),
            SimpleNamespace(type="text", text=" That is all.", citations=None),
        ],
    )
    calls = []
    monkeypatch.setattr(
        generator, "get_client", lambda: fake_client(response, calls)
    )

    result = generator.generate_answer("Laptop rules?", CHUNKS)

    assert result["answer"] == "Laptops need approved security software. That is all."
    assert result["citations"] == [
        {
            "cited_text": CHUNKS[0]["text"],
            "source": "it_policy.txt",
            "chunk_id": 0,
        }
    ]

    content = calls[0]["messages"][0]["content"]
    assert [block["title"] for block in content[:-1]] == [
        "it_policy.txt",
        "leave_policy.pdf",
    ]
    assert content[-1] == {"type": "text", "text": "Laptop rules?"}


def test_refusal(monkeypatch):
    response = SimpleNamespace(stop_reason="refusal", content=[])
    monkeypatch.setattr(
        generator, "get_client", lambda: fake_client(response, [])
    )

    result = generator.generate_answer("Anything", CHUNKS)

    assert result["citations"] == []
    assert result["answer"]
