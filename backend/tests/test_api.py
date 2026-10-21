from fastapi.testclient import TestClient

from app import main


def upload(client, path):
    with path.open("rb") as file:
        return client.post("/ingest", files={"file": (path.name, file)})


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_query_before_ingest(client):
    response = client.post("/query", json={"question": "Leave policy?"})

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_ingest_and_query(client, docs_dir, tmp_path):
    response = upload(client, docs_dir / "company_it_policy.txt")

    assert response.status_code == 200
    assert response.json()["chunks"] > 0
    assert (tmp_path / "vectorstore" / "index.faiss").exists()

    duplicate = upload(client, docs_dir / "company_it_policy.txt")
    assert duplicate.status_code == 409

    response = client.post(
        "/query",
        json={"question": "How do I protect my laptop?", "k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "stub answer"
    assert len(body["sources"]) == 2
    assert body["sources"][0]["source"] == "company_it_policy.txt"


def test_store_reloads_on_startup(client, docs_dir):
    upload(client, docs_dir / "leave_policy.pdf")

    # Restart the app: the lifespan should load the saved store.
    with TestClient(main.app) as restarted:
        response = restarted.post("/query", json={"question": "Annual leave?"})

    assert response.json()["sources"][0]["source"] == "leave_policy.pdf"


def test_rejects_unsupported_file(client, tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    assert upload(client, path).status_code == 400


def test_rejects_blank_question(client):
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422


def test_documents_lists_sources_with_chunk_counts(client, docs_dir):
    assert client.get("/documents").json() == []

    upload(client, docs_dir / "company_it_policy.txt")

    assert client.get("/documents").json() == [
        {"source": "company_it_policy.txt", "chunks": 2}
    ]


def test_missing_api_key_returns_503(client, docs_dir, monkeypatch):
    upload(client, docs_dir / "company_it_policy.txt")
    monkeypatch.setattr(main, "has_credentials", lambda: False)

    assert client.get("/health").json()["llm_configured"] is False

    for path, body in [("/query", {"question": "VPN?"}), ("/chat", {"message": "VPN?"})]:
        response = client.post(path, json=body)
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]

