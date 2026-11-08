import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from database import (
    get_all_documents,
    get_document_version,
    mark_document_active,
    mark_document_deleted,
    update_document_version,
)
from search_documents import embeddings_model, load_and_split_documents

DISTANCE_THRESHOLD = 0.90

# Connect to the existing ChromaDB collection
vector_store = Chroma(
    collection_name="python_basics",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)


def generate_chunk_id(chunk):
    unique_content = (
        chunk.metadata["document_id"]
        + str(chunk.metadata["document_version"])
        + chunk.page_content
    )

    return hashlib.sha256(unique_content.encode("utf-8")).hexdigest()


def add_new_chunks():
    chunks = load_and_split_documents()

    # Generate deterministic IDs for all current chunks
    chunk_ids = [generate_chunk_id(chunk) for chunk in chunks]

    # Retrieve existing ChromaDB records
    existing_records = vector_store.get(include=["metadatas"])
    existing_ids = set(existing_records["ids"])

    # Add missing chunk_id metadata to older records
    for chunk_id, metadata in zip(
        existing_records["ids"],
        existing_records["metadatas"],
    ):
        if "chunk_id" not in metadata:
            vector_store._collection.update(
                ids=[chunk_id],
                metadatas=[{**metadata, "chunk_id": chunk_id}],
            )

    # Prepare chunks that are not already stored
    new_chunks = []
    new_chunk_ids = []

    for chunk, chunk_id in zip(chunks, chunk_ids):
        chunk.metadata["chunk_id"] = chunk_id

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)

    # Insert new chunks without deleting old ones
    if new_chunks:
        vector_store.add_documents(
            documents=new_chunks,
            ids=new_chunk_ids,
        )

        print(f"Inserted {len(new_chunks)} new chunks.")
    else:
        print("No new chunks to insert.")

    # Verify that all current chunks exist in ChromaDB
    if chunk_ids:
        verified_records = vector_store.get(ids=chunk_ids)
        verified_ids = set(verified_records["ids"])
    else:
        verified_ids = set()

    missing_ids = set(chunk_ids) - verified_ids

    if missing_ids:
        raise RuntimeError(
            f"Indexing verification failed. {len(missing_ids)} chunks are missing."
        )

    print("All current chunks verified successfully.")

    # Retrieve all registered documents
    all_documents = get_all_documents()

    # Identify documents that successfully produced chunks
    updated_documents = {}

    for chunk in chunks:
        metadata = chunk.metadata

        updated_documents[metadata["document_id"]] = {
            "version": metadata["document_version"],
            "content_hash": metadata["content_hash"],
        }

    active_document_ids = set(updated_documents.keys())

    # Protect chunks belonging to existing active files that produced no chunks
    protected_document_ids = {
        document_id
        for document_id, source, status in all_documents
        if status == "ACTIVE"
        and Path(source).exists()
        and document_id not in active_document_ids
    }

    protected_ids = {
        chunk_id
        for chunk_id, metadata in zip(
            existing_records["ids"],
            existing_records["metadatas"],
        )
        if metadata.get("document_id") in protected_document_ids
    }

    # Identify outdated chunks, excluding protected chunks
    current_ids = set(chunk_ids)
    outdated_ids = existing_ids - current_ids - protected_ids

    # Delete outdated chunks only after successful verification
    if outdated_ids:
        vector_store.delete(ids=list(outdated_ids))
        print(f"Deleted {len(outdated_ids)} outdated chunks.")
    else:
        print("No outdated chunks found.")

    # Synchronize document versions and hashes
    for document_id, details in updated_documents.items():
        current_version = get_document_version(document_id)

        if current_version < details["version"]:
            update_document_version(
                document_id=document_id,
                document_version=details["version"],
                content_hash=details["content_hash"],
            )

        # Mark successfully indexed documents as active
        mark_document_active(document_id)

    # Mark documents whose source files no longer exist as deleted
    for document_id, source, status in all_documents:
        if status == "ACTIVE" and document_id not in active_document_ids:
            if not Path(source).exists():
                mark_document_deleted(document_id)
                print(f"Marked document as deleted: {source}")

    print("Document versions synchronized successfully.")
    print("Document statuses synchronized successfully.")


def show_stored_documents():
    stored_documents = vector_store.get(include=["metadatas"])

    for index, metadata in enumerate(stored_documents["metadatas"]):
        print(f"\nChunk {index + 1}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'MISSING')}")
        print(f"Document ID: {metadata['document_id']}")
        print(f"Document name: {metadata['document_name']}")
        print(f"Document version: {metadata['document_version']}")
        print(f"Source: {metadata['source']}")
        print(f"File type: {metadata['file_type']}")
        print(f"Page number: {metadata.get('page_number', 'N/A')}")


if __name__ == "__main__":
    add_new_chunks()
    show_stored_documents()

    print("Total documents:", vector_store._collection.count())

    question = "What is the capital of Testland?"

    results = vector_store.similarity_search_with_score(
        query=question,
        k=3,
    )

    relevant_documents = [
        (document, distance)
        for document, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    print("\nRelevant documents:")

    if not relevant_documents:
        print("No relevant information found in the document.")

    else:
        for index, (document, distance) in enumerate(
            relevant_documents,
            start=1,
        ):
            print(f"\nResult {index}:")
            print(f"Distance score: {distance:.4f}")
            print(document.page_content)
