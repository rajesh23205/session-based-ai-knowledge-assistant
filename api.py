from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chat_service

app = FastAPI(
    title="Session-Based AI Knowledge Assistant",
    description="API for an AI-powered knowledge assistant",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1)


# Response models


class SessionResponse(BaseModel):
    session_id: str


# API endpoints


@app.get("/")
def home():
    return {"message": "AI Knowledge Assistant API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session():
    session_id = chat_service.create_new_session()

    return {"session_id": session_id}


@app.get("/sessions")
def get_sessions():
    return chat_service.list_sessions()


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = chat_service.get_session_details(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return session


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    deleted = chat_service.database.delete_session(session_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return {"message": "Session deleted successfully"}


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )

    result = chat_service.process_chat(
        request.session_id,
        question,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return result
