# Session-Based AI Knowledge Assistant

An AI-powered knowledge assistant built with Python, FastAPI, React, TypeScript, SQLite, ChromaDB, LangChain, and OpenAI.

The application allows users to maintain multiple conversation sessions and ask questions grounded in uploaded knowledge-base documents.

## Features

- Session-based conversations
- Persistent conversation history using SQLite
- Context-aware question rewriting
- Conversation summarization and memory management
- Document-based Retrieval-Augmented Generation (RAG)
- PDF and TXT document processing
- Text chunking and OpenAI embeddings
- Semantic similarity search using ChromaDB
- Top-K retrieval with similarity filtering
- Source attribution with document and page metadata
- Document version tracking and synchronization
- Grounded answer generation and claim validation
- Unsupported-question handling and clarification
- FastAPI backend
- React and TypeScript frontend
- Automated backend API and source-persistence tests

## Tech Stack

**Backend**
- Python 3.13
- FastAPI
- LangChain
- OpenAI
- ChromaDB
- SQLite

**Frontend**
- React
- TypeScript
- Vite
- CSS

**Testing**
- Pytest
- HTTPX
- FastAPI TestClient

## Project Structure

```text
session-based-ai-knowledge-assistant/
│
├── documents/
├── frontend/
├── tests/
│
├── api.py
├── chat_service.py
├── context_manager.py
├── database.py
├── rag_answer.py
├── search_documents.py
├── vector_store.py
├── message_demo.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/rajesh23205/session-based-ai-knowledge-assistant.git

cd session-based-ai-knowledge-assistant
```

### 2. Create a virtual environment

```bash
py -3.13 -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 5. Start the backend

```bash
uvicorn api:app --reload
```

Backend: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

### 6. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

## Running Tests

From the project root:

```bash
python -m pytest -v
```

## Architecture Overview

1. Users create and manage conversation sessions.
2. Questions are sent to the FastAPI backend.
3. Conversation history and summaries provide contextual information.
4. Questions are rewritten when necessary to make them independently searchable.
5. ChromaDB retrieves relevant document chunks using semantic similarity.
6. The LLM generates an answer grounded in retrieved documents.
7. Answer claims are validated against retrieved context.
8. Answers and source metadata are persisted in SQLite.
9. The frontend displays answers and their supporting sources.

## Future Improvements

- User authentication and authorization
- Document upload interface
- Streaming responses
- Advanced LangGraph workflows
- Evaluation and observability
- Production deployment

## Author

Built as a hands-on project to explore RAG systems, conversational memory, AI agents, and full-stack AI application development.
