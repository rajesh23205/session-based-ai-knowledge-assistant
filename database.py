import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "conversation_history.db"


@contextmanager
def get_connection():
    """
    Create a database connection for one operation.
    Automatically commit, rollback, and close the connection.
    """

    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
            )
        """)

        # Add source storage to existing messages tables.
        columns = connection.execute("PRAGMA table_info(messages)").fetchall()

        column_names = [column[1] for column in columns]

        if "sources_json" not in column_names:
            connection.execute("""
                ALTER TABLE messages
                ADD COLUMN sources_json TEXT
            """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                source TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                content_hash TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'ACTIVE'
            )
        """)

        columns = connection.execute("PRAGMA table_info(documents)").fetchall()

        column_names = [column[1] for column in columns]

        if "status" not in column_names:
            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'
            """)


# Session management


def create_session(session_id):
    with get_connection() as connection:
        result = connection.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
            (session_id,),
        )

        return result.rowcount == 1


def session_exists(session_id):
    with get_connection() as connection:
        result = connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        return result is not None


def get_all_sessions():
    with get_connection() as connection:
        return connection.execute("""
            SELECT session_id, created_at
            FROM sessions
            ORDER BY created_at DESC
        """).fetchall()


def delete_session(session_id):
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )

        connection.execute(
            "DELETE FROM conversation_summaries WHERE session_id = ?",
            (session_id,),
        )

        result = connection.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )

        return result.rowcount == 1


# Conversation messages


def save_message(session_id, role, content, sources=None):
    sources_json = json.dumps(sources) if sources else None

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages (
                session_id,
                role,
                content,
                sources_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, sources_json),
        )


def get_messages(session_id):
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()


def get_messages_with_sources(session_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources_json
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    messages = []

    for role, content, sources_json in rows:
        messages.append(
            {
                "role": role,
                "content": content,
                "sources": json.loads(sources_json) if sources_json else [],
            }
        )

    return messages


def load_conversation_history(session_id):
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
    )

    messages = get_messages(session_id)
    conversation_history = []

    for role, content in messages:
        if role == "human":
            conversation_history.append(HumanMessage(content=content))

        elif role == "ai":
            conversation_history.append(AIMessage(content=content))

        elif role == "system":
            conversation_history.append(SystemMessage(content=content))

    return conversation_history


# Conversation summaries


def save_summary(session_id, summary):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversation_summaries (session_id, summary)
            VALUES (?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET
                summary = excluded.summary,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, summary),
        )


def get_summary(session_id):
    with get_connection() as connection:
        result = connection.execute(
            """
            SELECT summary
            FROM conversation_summaries
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        return result[0] if result else None


# Document management


def get_document(source):
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT
                document_id,
                document_name,
                file_type,
                content_hash,
                version,
                status
            FROM documents
            WHERE source = ?
            """,
            (source,),
        ).fetchone()


def register_document(document_name, source, file_type, content_hash):
    result = get_document(source)

    if result:
        return result[0]

    document_id = f"doc_{uuid.uuid4()}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id,
                document_name,
                source,
                file_type,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, document_name, source, file_type, content_hash),
        )

    return document_id


def update_document_version(document_id, document_version, content_hash):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE documents
            SET content_hash = ?,
                version = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE document_id = ?
            """,
            (content_hash, document_version, document_id),
        )


def mark_document_deleted(document_id):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE documents
            SET status = 'DELETED',
                updated_at = CURRENT_TIMESTAMP
            WHERE document_id = ?
            """,
            (document_id,),
        )


def get_all_documents():
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT document_id, source, status
            FROM documents
            """
        ).fetchall()


def mark_document_active(document_id):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE documents
            SET status = 'ACTIVE',
                updated_at = CURRENT_TIMESTAMP
            WHERE document_id = ?
            """,
            (document_id,),
        )


def get_document_version(document_id):
    with get_connection() as connection:
        result = connection.execute(
            """
            SELECT version
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        return result[0] if result else None


# Initialize database tables when this module is imported

initialize_database()
