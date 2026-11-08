# Session-Based AI Knowledge Assistant

A Python-based AI knowledge assistant that combines conversational memory, session management, persistent storage, and Retrieval-Augmented Generation (RAG) to provide context-aware answers from documents.

## Features

### 1. Conversation History
- Maintains conversation history using LangChain messages.
- Supports `HumanMessage`, `AIMessage`, and `SystemMessage`.
- Uses conversation history to understand follow-up questions.

### 2. Session Management
- Supports multiple independent conversation sessions.
- Maintains separate conversation history for each session.
- Allows users to resume existing sessions.

### 3. Persistent Conversation History
- Uses SQLite to store conversation messages.
- Restores conversation history when an existing session resumes.
- Stores conversation summaries separately from individual messages.

### 4. Context-Aware RAG
- Loads TXT and PDF documents from the documents directory.
- Splits documents into smaller chunks.
- Generates embeddings using OpenAI embeddings.
- Stores and retrieves document chunks using ChromaDB.
- Uses similarity search to retrieve relevant information.
- Rewrites follow-up questions into standalone questions.
- Generates answers grounded in retrieved document context.
- Returns source information with answers.
- Handles questions when relevant information is unavailable.

### 5. Conversation Context Management
- Counts tokens using `tiktoken`.
- Applies a configurable conversation history token budget.
- Summarizes older conversation turns automatically.
- Updates existing summaries incrementally.
- Retains the latest three complete conversation turns.
- Preserves the previous summary if summarization fails.
- Saves updated summaries in SQLite.
- Uses conversation summaries and recent history to maintain continuity.

### 6. Sources and Reliability
- Tracks document metadata, including document ID, source, file type, version, and content hash.
- Preserves PDF page numbers in document chunks.
- Provides source attribution with retrieved answers.
- Validates generated answers against retrieved document content.
- Identifies supported, partially supported, and unsupported claims.
- Corrects partially supported answers to retain only supported information.
- Detects ambiguous follow-up questions and requests clarification before retrieval.
- Handles conflicting information by identifying the claims made by individual sources.
- Uses similarity thresholds to filter irrelevant retrieval results.
- Detects document content changes and maintains document versions.
- Identifies deleted documents and removes outdated chunks.
- Verifies newly indexed chunks before deleting outdated records.
- Protects existing indexed data when document processing produces no chunks.
- Includes tests for supported, unsupported, partially supported, ambiguous, and conflicting questions.

## Technologies Used

- Python
- LangChain
- OpenAI API
- OpenAI Embeddings (`text-embedding-3-small`)
- ChromaDB
- SQLite
- NumPy
- tiktoken
- python-dotenv
- pypdf

## Project Structure

```text
session-based-ai-knowledge-assistant/
│
├── documents/
│   ├── machine_learning.txt
│   └── machine_learning_guide.pdf
│
├── chroma_db/
│
├── message_demo.py
├── database.py
├── context_manager.py
├── search_documents.py
├── vector_store.py
├── rag_answer.py
├── test_rag.py
├── token_demo.py
│
├── requirements.txt
├── .env
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
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your Open