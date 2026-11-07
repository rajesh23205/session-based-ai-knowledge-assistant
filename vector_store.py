import hashlib

from langchain_chroma import Chroma

from search_documents import embeddings_model, load_and_split_documents

DISTANCE_THRESHOLD = 0.90

# Connect to the existing ChromaDB collection
vector_store = Chroma(
    collection_name="python_basics",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)


def generate_chunk_id(chunk):
    unique_content = chunk.metadata["source"] + chunk.page_content

    return hashlib.sha256(unique_content.encode("utf-8")).hexdigest()


def add_new_chunks():
    chunks = load_and_split_documents()

    chunk_ids = [generate_chunk_id(chunk) for chunk in chunks]

    # Get IDs of documents already stored in ChromaDB
    existing_ids = set(vector_store.get()["ids"])

    current_ids = set(chunk_ids)

    # Find chunks that exist in ChromaDB but not in current documents
    outdated_ids = existing_ids - current_ids

    if outdated_ids:
        vector_store.delete(ids=list(outdated_ids))
        print(f"Deleted {len(outdated_ids)} outdated chunks.")

    print("Outdated chunks:", len(outdated_ids))

    # Identify chunks that are not already stored
    new_chunks = []
    new_chunk_ids = []

    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)

    # Insert only new chunks
    if new_chunks:
        vector_store.add_documents(
            documents=new_chunks,
            ids=new_chunk_ids,
        )

        print(f"Added {len(new_chunks)} new chunks.")

    else:
        print("No new chunks to add.")


def show_stored_documents():
    stored_documents = vector_store.get(include=["metadatas"])

    for metadata in stored_documents["metadatas"]:
        print(metadata["source"])


if __name__ == "__main__":
    add_new_chunks()
    show_stored_documents()

    print("Total documents:", vector_store._collection.count())

    question = "What is supervised learning?"

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
        for index, (document, distance) in enumerate(relevant_documents, start=1):
            print(f"\nResult {index}:")
            print(f"Distance score: {distance:.4f}")
            print(document.page_content)
