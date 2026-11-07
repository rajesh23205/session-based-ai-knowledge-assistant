import sqlite3

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

connection.commit()

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

connection.commit()

# Create conversation summaries table
connection.execute("""
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        session_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
""")

connection.commit()


# Create a new session if it doesn't already exist
def create_session(session_id):
    result = connection.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    if result is None:
        print("Session does not exist. Creating a new session...")

        connection.execute(
            "INSERT INTO sessions (session_id) VALUES (?)", (session_id,)
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


if __name__ == "__main__":
    test_session = "summary_test"

    create_session(test_session)

    save_summary(test_session, "Alex is learning Python.")

    print("First summary:", get_summary(test_session))

    save_summary(test_session, "Alex is learning Python and OOP.")

    print("Updated summary:", get_summary(test_session))
