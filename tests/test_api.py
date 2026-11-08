import pytest
from fastapi.testclient import TestClient

import api
import chat_service
import database


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_create_session(client):
    response = client.post("/sessions")

    assert response.status_code == 201

    session_id = response.json()["session_id"]

    assert session_id.startswith("session_")
    assert database.session_exists(session_id)

    database.delete_session(session_id)


def test_get_sessions(client):
    session_id = chat_service.create_new_session()

    try:
        response = client.get("/sessions")

        assert response.status_code == 200

        session_ids = [session["session_id"] for session in response.json()]

        assert session_id in session_ids

    finally:
        database.delete_session(session_id)


def test_get_session(client):
    session_id = chat_service.create_new_session()

    try:
        response = client.get(f"/sessions/{session_id}")

        assert response.status_code == 200

        data = response.json()

        assert data["session_id"] == session_id
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "system"

    finally:
        database.delete_session(session_id)


def test_get_nonexistent_session(client):
    response = client.get("/sessions/nonexistent_session")

    assert response.status_code == 404


def test_delete_session(client):
    session_id = chat_service.create_new_session()

    response = client.delete(f"/sessions/{session_id}")

    assert response.status_code == 200
    assert not database.session_exists(session_id)


def test_delete_nonexistent_session(client):
    response = client.delete("/sessions/nonexistent_session")

    assert response.status_code == 404


def test_chat_with_nonexistent_session(client):
    response = client.post(
        "/chat",
        json={
            "session_id": "nonexistent_session",
            "question": "What is machine learning?",
        },
    )

    assert response.status_code == 404


def test_chat_success(client, monkeypatch):
    session_id = chat_service.create_new_session()

    expected_result = {
        "session_id": session_id,
        "question": "What is machine learning?",
        "answer": "Machine learning allows computers to learn from data.",
        "sources": [],
        "standalone_question": "What is machine learning?",
        "validation": {
            "status": "SUPPORTED",
            "supported_claims": [
                "Machine learning allows computers to learn from data."
            ],
            "unsupported_claims": [],
            "corrected_answer": "Machine learning allows computers to learn from data.",
        },
        "needs_clarification": False,
    }

    def mock_process_chat(session_id, question):
        return expected_result

    monkeypatch.setattr(
        chat_service,
        "process_chat",
        mock_process_chat,
    )

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "question": "What is machine learning?",
            },
        )

        assert response.status_code == 200
        assert response.json() == expected_result

    finally:
        database.delete_session(session_id)


def test_chat_rejects_empty_question(client):
    session_id = chat_service.create_new_session()

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "question": "",
            },
        )

        assert response.status_code == 422

    finally:
        database.delete_session(session_id)


def test_chat_rejects_whitespace_question(client):
    session_id = chat_service.create_new_session()

    try:
        response = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "question": "   ",
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Question cannot be empty"

    finally:
        database.delete_session(session_id)
