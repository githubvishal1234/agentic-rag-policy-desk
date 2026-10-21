from app.services.ingestion import ingest_file


def test_embeddings(docs_dir, embedding_service):
    chunks = ingest_file(
        str(docs_dir / "company_it_policy.txt"),
        chunk_size=200,
        chunk_overlap=40,
    )

    vectors = embedding_service.embed_documents(chunks)

    assert len(vectors) == len(chunks)
    assert all(len(vector) == 384 for vector in vectors)

    query_vector = embedding_service.embed_query(
        "What are the rules for company laptop security?"
    )

    assert len(query_vector) == 384
