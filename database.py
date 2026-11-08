import hashlib
import sqlite3
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Connect to SQLite database
connection = sqlite3.connect("conversation_history.db")

# Enable foreign key enforcement
connection.execute("PRAGMA foreign_keys = ON")


# Create sessions table
connection.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Create messages table
connection.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
""")

# Create conversation summaries table
connection.execute("""
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        session_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
""")


# Create documents table
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


# Safely migrate existing documents table
columns = connection.execute("PRAGMA table_info(documents)").fetchall()

column_names = [column[1] for column in columns]

if "status" not in column_names:
    connection.execute("""
        ALTER TABLE documents
        ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'
    """)

connection.commit()


# Create a new session if it doesn't already exist
def create_session(session_id):
    result = connection.execute(
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    if result is None:
        print("Session does not exist. Creating a new session...")

        connection.execute(
            "INSERT INTO sessions (session_id) VALUES (?)",
            (session_id,),
        )

        connection.commit()

        print("Session created successfully!")

        return True

    else:
        print("Session already exists.")

        return False


# Save a conversation message
def save_message(session_id, role, content):
    connection.execute(
        """
        INSERT INTO messages (session_id, role, content)
        VALUES (?, ?, ?)
        """,
        (session_id, role, content),
    )

    connection.commit()


# Save or update a conversation summary
def save_summary(session_id, summary):
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

    connection.commit()


# Retrieve a conversation summary
def get_summary(session_id):
    result = connection.execute(
        """
        SELECT summary
        FROM conversation_summaries
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    return result[0] if result else None


# Retrieve conversation messages
def get_messages(session_id):
    result = connection.execute(
        """
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()

    return result


# Reconstruct LangChain conversation history
def load_conversation_history(session_id):
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


# Retrieve document metadata
def get_document(source):
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


# Register a new document


def register_document(document_name, source, file_type, content_hash):
    result = get_document(source)

    if result:
        return result[0]
    document_id = f"doc_{uuid.uuid4()}"

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

    connection.commit()

    return document_id


# Update document version and content hash
def update_document_version(document_id, document_version, content_hash):
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

    connection.commit()


def mark_document_deleted(document_id):
    connection.execute(
        """
        UPDATE documents
        SET status = 'DELETED',
            updated_at = CURRENT_TIMESTAMP
        WHERE document_id = ?
        """,
        (document_id,),
    )

    connection.commit()


def get_all_documents():
    return connection.execute(
        """
        SELECT document_id, source, status
        FROM documents
        """
    ).fetchall()


def mark_document_active(document_id):
    connection.execute(
        """
        UPDATE documents
        SET status = 'ACTIVE',
            updated_at = CURRENT_TIMESTAMP
        WHERE document_id = ?
        """,
        (document_id,),
    )

    connection.commit()


# Retrieve document version
def get_document_version(document_id):
    result = connection.execute(
        "SELECT version FROM documents WHERE document_id = ?",
        (document_id,),
    ).fetchone()

    return result[0] if result else None


if __name__ == "__main__":
    test_session = "summary_test"

    create_session(test_session)

    save_summary(test_session, "Alex is learning Python.")

    print("First summary:", get_summary(test_session))

    save_summary(test_session, "Alex is learning Python and OOP.")

    print("Updated summary:", get_summary(test_session))

    document_path = "documents/machine_learning.txt"

    with open(document_path, "rb") as file:
        content_hash = hashlib.sha256(file.read()).hexdigest()

    document_id = register_document(
        document_name="machine_learning.txt",
        source=document_path,
        file_type="txt",
        content_hash=content_hash,
    )

    print("Document ID:", document_id)

    # Test document version management
    test_source = "documents/registration_test.txt"

    original_hash = hashlib.sha256(b"Original content").hexdigest()
    updated_hash = hashlib.sha256(b"Updated content").hexdigest()

    test_id = register_document(
        "registration_test.txt",
        test_source,
        "txt",
        original_hash,
    )

    register_document(
        "registration_test.txt",
        test_source,
        "txt",
        updated_hash,
    )

    result = connection.execute(
        """
        SELECT document_id, version
        FROM documents
        WHERE source = ?
        """,
        (test_source,),
    ).fetchone()

    print("Test document ID:", result[0])
    print("Test document version:", result[1])

    # Clean up test record
    connection.execute(
        "DELETE FROM documents WHERE source = ?",
        (test_source,),
    )

    connection.commit()
