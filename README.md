# Session-Based AI Knowledge Assistant

A Python-based AI knowledge assistant that combines conversational memory, session management, persistent storage, and Retrieval-Augmented Generation (RAG) to provide context-aware answers from documents.

## Features

### 1. Conversation History
- Maintains conversation history using LangChain messages.
- Supports HumanMessage, AIMessage, and SystemMessage.
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
- Loads documents from the documents directory.
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

## Project Structure

```text
session-based-ai-knowledge-assistant/
│
├── documents/
│   └── machine_learning.txt
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

Replace the placeholder with your OpenAI API key.

## Run the Application

```bash
python message_demo.py
```

Enter a session ID to start a new conversation or resume an existing session.

Type `exit` to end the conversation.

## Project Roadmap

- [x] Phase 1: Conversation History
- [x] Phase 2: Session Management
- [x] Phase 3: Persistent Conversation History
- [x] Phase 4: Context-Aware RAG
- [x] Phase 5: Conversation Context Management
- [ ] Phase 6: Sources and Reliability
- [ ] Phase 7: Chat UI and Backend
- [ ] Phase 8: Testing and Improvements

## Learning Objective

This project is being developed step by step to understand how AI-powered knowledge assistants work, including conversational memory, session isolation, persistent storage, embeddings, vector search, RAG, token management, and conversation summarization.
```
