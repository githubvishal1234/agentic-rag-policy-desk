import pytest

from app.services.ingestion import ingest_file
from app.services.vector_store import FaissVectorStore


@pytest.mark.parametrize(
    "query",
    [
        "How should employees protect company laptops?",
        "What are the password rules?",
        "What should employees do if a device is stolen?",
    ],
)
def test_similarity_search(docs_dir, embedding_service, query):
    chunks = ingest_file(
        str(docs_dir / "company_it_policy.txt"),
        chunk_size=200,
        chunk_overlap=40,
    )

    vector_store = FaissVectorStore(embedding_service)
    vector_store.add_documents(chunks)

    results = vector_store.similarity_search(query, k=2)

    assert len(results) == 2, f"Expected 2 results for: {query}"

    for result in results:
        assert result["metadata"]["source"] == "company_it_policy.txt"


@pytest.fixture
def all_samples_store(docs_dir, embedding_service):
    vector_store = FaissVectorStore(embedding_service)

    for path in sorted(docs_dir.iterdir()):
        vector_store.add_documents(
            ingest_file(str(path), chunk_size=200, chunk_overlap=40)
        )

    return vector_store


def test_hybrid_search_finds_keyword_match(all_samples_store):
    # Vector search alone ranks the password chunk first for this query.
    results = all_samples_store.hybrid_search("lost laptop", k=1)

    assert "lost or stolen devices" in results[0]["text"]


def test_hybrid_search_without_keyword_matches(all_samples_store):
    # No query word appears in any document, so only vector results remain.
    results = all_samples_store.hybrid_search("smartphone theft", k=3)

    assert len(results) == 3


def test_hybrid_search_reloads_keyword_index(all_samples_store, tmp_path):
    all_samples_store.save(str(tmp_path))

    restored = FaissVectorStore(all_samples_store.embedding_service)
    restored.load(str(tmp_path))

    query = "lost laptop"
    assert restored.hybrid_search(query) == all_samples_store.hybrid_search(query)


def test_remove_source(all_samples_store):
    before = len(all_samples_store.documents)
    it_chunks = sum(
        d.metadata["source"] == "company_it_policy.txt"
        for d in all_samples_store.documents
    )

    assert all_samples_store.remove_source("company_it_policy.txt") == it_chunks
    assert all_samples_store.remove_source("company_it_policy.txt") == 0
    assert all_samples_store.index.ntotal == len(all_samples_store.documents) == before - it_chunks

    # Remaining vectors still line up with their texts after compaction.
    results = all_samples_store.hybrid_search("annual leave", k=4)
    assert all(r["metadata"]["source"] != "company_it_policy.txt" for r in results)
    assert "leave" in results[0]["text"].lower()
    top = all_samples_store.similarity_search(results[0]["text"], k=1)[0]
    assert top["score"] > 0.99


def test_remove_last_source_empties_store(docs_dir, embedding_service):
    store = FaissVectorStore(embedding_service)
    store.add_documents(ingest_file(str(docs_dir / "company_it_policy.txt")))

    store.remove_source("company_it_policy.txt")

    assert store.index is None
    assert store.hybrid_search("VPN") == []
