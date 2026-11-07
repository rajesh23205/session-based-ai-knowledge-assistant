from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# Initialize the embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


# Step 1: Load and split the document
def load_and_split_documents():
    documents = []

    document_paths = Path("documents").glob("*.txt")

    for document_path in document_paths:
        document_text = document_path.read_text(encoding="utf-8")

        document = Document(
            page_content=document_text, metadata={"source": str(document_path)}
        )

        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    return text_splitter.split_documents(documents)


# Run document loading when this file is executed directly
if __name__ == "__main__":
    # Load document and generate embeddings
    chunks = load_and_split_documents()

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\nChunk index: {index}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Content: {chunk.page_content}")
