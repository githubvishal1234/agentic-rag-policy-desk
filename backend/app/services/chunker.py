
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into overlapping, character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    langchain_documents = splitter.create_documents(
        texts=[document.text for document in documents],
        metadatas=[document.metadata for document in documents],
    )

    chunks = []

    for chunk_id, chunk in enumerate(langchain_documents):
        metadata = chunk.metadata.copy()
        metadata["chunk_id"] = chunk_id

        chunks.append(
            Document(
                text=chunk.page_content,
                metadata=metadata,
            )
        )

    return chunks
