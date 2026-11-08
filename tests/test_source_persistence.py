import chat_service
import database


def test_message_sources_persist_after_loading_session():
    session_id = chat_service.create_new_session()

    expected_sources = [
        {
            "chunk_id": "chunk_test_001",
            "document_id": "doc_test_001",
            "document_name": "machine_learning_guide.pdf",
            "document_version": 1,
            "file_type": "pdf",
            "source": "documents/machine_learning_guide.pdf",
            "page_number": 3,
            "distance": 0.25,
            "content": "Machine learning enables computers to learn from data.",
        }
    ]

    try:
        database.save_message(
            session_id,
            "ai",
            "Machine learning enables computers to learn from data.",
            sources=expected_sources,
        )

        session = chat_service.get_session_details(session_id)

        assert session is not None

        ai_message = session["messages"][1]

        assert ai_message["role"] == "ai"
        assert ai_message["sources"] == expected_sources

    finally:
        database.delete_session(session_id)


def test_old_messages_without_sources_remain_compatible():
    session_id = chat_service.create_new_session()

    try:
        database.save_message(
            session_id,
            "ai",
            "This is an older message.",
        )

        session = chat_service.get_session_details(session_id)

        ai_message = session["messages"][1]

        assert ai_message["content"] == "This is an older message."
        assert ai_message["sources"] == []

    finally:
        database.delete_session(session_id)
