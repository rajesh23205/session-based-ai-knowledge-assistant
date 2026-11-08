import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from database import get_document, register_document

load_dotenv()


# Initialize the embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


# Load and split documents
def load_and_split_documents():
    documents = []

    document_paths = [
        *Path("documents").glob("*.txt"),
        *Path("documents").glob("*.pdf"),
    ]

    for document_path in document_paths:
        # Normalize the document path to prevent duplicate records
        source = os.path.normcase(os.path.normpath(str(document_path)))

        # Read the original document as bytes
        document_bytes = document_path.read_bytes()

        # Generate a SHA-256 content hash
        content_hash = hashlib.sha256(document_bytes).hexdigest()

        # Extract content based on file type
        if document_path.suffix.lower() == ".txt":
            document_text = document_bytes.decode("utf-8")

            pages = [
                {
                    "page_content": document_text,
                    "page_number": None,
                }
            ]

        elif document_path.suffix.lower() == ".pdf":
            pdf_reader = PdfReader(str(document_path))

            pages = [
                {
                    "page_content": page.extract_text() or "",
                    "page_number": page_number,
                }
                for page_number, page in enumerate(pdf_reader.pages, start=1)
            ]

        else:
            continue

        # Retrieve existing document metadata using normalized path
        existing_document = get_document(source)

        if existing_document:
            (
                document_id,
                _,
                _,
                stored_hash,
                current_version,
                status,
            ) = existing_document

            if stored_hash == content_hash:
                document_version = current_version
            else:
                document_version = current_version + 1

        else:
            document_id = register_document(
                document_name=document_path.name,
                source=source,
                file_type=document_path.suffix.lstrip("."),
                content_hash=content_hash,
            )

            document_version = 1

        # Create a LangChain document for each page
        for page in pages:
            if not page["page_content"].strip():
                continue

            document = Document(
                page_content=page["page_content"],
                metadata={
                    "source": source,
                    "document_id": document_id,
                    "document_name": document_path.name,
                    "file_type": document_path.suffix.lstrip("."),
                    "document_version": document_version,
                    "content_hash": content_hash,
                    "page_number": page["page_number"],
                },
            )

            documents.append(document)

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    return text_splitter.split_documents(documents)


# Run document loading when this file is executed directly
if __name__ == "__main__":
    chunks = load_and_split_documents()

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):
        print(f"\nChunk index: {index}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Content: {chunk.page_content}")
        print(f"Document ID: {chunk.metadata['document_id']}")
        print(f"Document name: {chunk.metadata['document_name']}")
        print(f"Document version: {chunk.metadata['document_version']}")
        print(f"File type: {chunk.metadata['file_type']}")
        print(f"Page number: {chunk.metadata['page_number']}")
