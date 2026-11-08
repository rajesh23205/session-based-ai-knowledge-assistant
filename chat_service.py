import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

import database
from context_manager import manage_conversation_context, split_conversation_history
from rag_answer import generate_answer

load_dotenv()


SYSTEM_PROMPT = (
    "You are a helpful AI knowledge assistant. "
    "Answer questions using the provided documents. "
    "Do not invent information."
)


def create_new_session():
    """Create a session and initialize its system message."""

    session_id = f"session_{uuid.uuid4().hex[:12]}"

    database.create_session(session_id)
    database.save_message(session_id, "system", SYSTEM_PROMPT)

    return session_id


def list_sessions():
    """Return all available sessions."""

    sessions = database.get_all_sessions()

    return [
        {
            "session_id": session_id,
            "created_at": created_at,
        }
        for session_id, created_at in sessions
    ]


def get_session_details(session_id):
    """Return a session's conversation history."""

    if not database.session_exists(session_id):
        return None

    messages = database.get_messages_with_sources(session_id)

    return {
        "session_id": session_id,
        "messages": messages,
    }


def process_chat(session_id, question):
    """Process a question using conversation context and document retrieval."""

    if not database.session_exists(session_id):
        return None

    conversation_history = database.load_conversation_history(session_id)
    summary = database.get_summary(session_id)

    if summary:
        _, conversation_history = split_conversation_history(conversation_history)

    previous_summary = summary

    summary, request_history = manage_conversation_context(
        conversation_history,
        existing_summary=summary,
    )

    if summary and summary != previous_summary:
        database.save_summary(session_id, summary)

    conversation_history = request_history

    result = generate_answer(
        question,
        conversation_history,
        summary,
    )

    # Save the user's question.
    human_message = HumanMessage(content=question)
    conversation_history.append(human_message)

    database.save_message(session_id, "human", question)

    # Save the assistant's response.
    ai_message = AIMessage(content=result["answer"])
    conversation_history.append(ai_message)

    database.save_message(
        session_id,
        "ai",
        ai_message.content,
        sources=result["sources"],
    )

    return {
        "session_id": session_id,
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "standalone_question": result["standalone_question"],
        "validation": result["validation"],
        "needs_clarification": result["needs_clarification"],
    }
