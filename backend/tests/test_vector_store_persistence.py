from app.services.ingestion import ingest_file
from app.services.vector_store import FaissVectorStore


def test_save_and_load(docs_dir, embedding_service, tmp_path):
    chunks = ingest_file(
        str(docs_dir / "company_it_policy.txt"),
        chunk_size=200,
        chunk_overlap=40,
    )

    original_store = FaissVectorStore(embedding_service)
    original_store.add_documents(chunks)

    query = "What should employees do if a device is stolen?"
    original_results = original_store.similarity_search(query, k=2)

    original_store.save(str(tmp_path))

    restored_store = FaissVectorStore(embedding_service)
    restored_store.load(str(tmp_path))

    restored_results = restored_store.similarity_search(query, k=2)

    assert len(restored_results) == len(original_results)

    for original, restored in zip(original_results, restored_results):
        assert original["text"] == restored["text"]
        assert original["metadata"] == restored["metadata"]
        assert abs(original["score"] - restored["score"]) < 1e-5
