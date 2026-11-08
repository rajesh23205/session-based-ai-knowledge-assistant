from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import database
from context_manager import manage_conversation_context, split_conversation_history
from rag_answer import generate_answer

load_dotenv()

# Select or create a session
session_id = input("Enter session ID: ").strip()

if not session_id:
    raise ValueError("Session ID cannot be empty.")

is_new_session = database.create_session(session_id)

# Load or initialize conversation history
if is_new_session:
    print("New session created!")

    system_message = SystemMessage(
        content=(
            "You are a helpful AI knowledge assistant. "
            "Answer questions using the provided documents. "
            "Do not invent information."
        )
    )

    conversation_history = [system_message]
    summary = None
    database.save_message(session_id, "system", system_message.content)

else:
    print("Existing session found!")

    conversation_history = database.load_conversation_history(session_id)
    summary = database.get_summary(session_id)

    if summary:
        _, conversation_history = split_conversation_history(conversation_history)

# Interactive conversation loop
while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Ending conversation...")
        break

    if not question:
        print("Please enter a question.")
        continue

    # Generate an answer using previous conversation history
    previous_summary = summary

    summary, request_history = manage_conversation_context(
        conversation_history,
        existing_summary=summary,
    )

    if summary and summary != previous_summary:
        database.save_summary(session_id, summary)

    conversation_history = request_history

    result = generate_answer(question, conversation_history, summary)

    # Add and save user's question
    human_message = HumanMessage(content=question)
    conversation_history.append(human_message)
    summary = None

    database.save_message(session_id, "human", question)

    # Add and save AI response
    ai_message = result["answer"]

    response = AIMessage(content=ai_message)

    conversation_history.append(response)

    database.save_message(session_id, "ai", response.content)

    print("\nAI:", response.content)

    print("\nSources:")

    if result["sources"]:
        for index, source in enumerate(result["sources"], start=1):
            print(f"\nSource {index}:")
            print(f"Document: {source['document_name']}")

            if source["page_number"] is not None:
                print(f"Page: {source['page_number']}")

            print(f"Chunk ID: {source['chunk_id']}")
            print(f"Document version: {source['document_version']}")
            print(f"Distance score: {source['distance']:.4f}")
            print(f"Content: {source['content']}")

    else:
        print("No relevant document sources found.")

# Inspect the selected session
print(f"\n--- {session_id} Conversation History ---")

for index, message in enumerate(conversation_history, start=1):
    print(f"{index}. {type(message).__name__}: {message.content[:60]}")
