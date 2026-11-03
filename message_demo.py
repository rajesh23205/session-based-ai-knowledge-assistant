from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import database

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Select or create a session
session_id = input("Enter session ID: ").strip()

is_new_session = database.create_session(session_id)

# Load or initialize conversation history
if is_new_session:
    print("New session created!")

    system_message = SystemMessage(content="You are a helpful AI programming tutor.")

    conversation_history = [system_message]

    database.save_message(session_id, "system", system_message.content)

else:
    print("Existing session found!")

    conversation_history = database.load_conversation_history(session_id)

# Interactive conversation loop
while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Ending conversation...")
        break

    if not question:
        print("Please enter a question.")
        continue

    # Add and save user's question
    human_message = HumanMessage(content=question)
    conversation_history.append(human_message)

    database.save_message(session_id, "human", question)

    # Generate AI response using conversation history
    response = llm.invoke(conversation_history)

    # Add and save AI response
    conversation_history.append(response)

    database.save_message(session_id, "ai", response.content)

    print("\nAI:", response.content)

# Inspect the selected session
print(f"\n--- {session_id} Conversation History ---")

for index, message in enumerate(conversation_history, start=1):
    print(f"{index}. {type(message).__name__}: {message.content[:60]}")
