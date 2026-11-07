import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

MAX_HISTORY_TOKENS = 1000
SUMMARY_TRIGGER = 10
RECENT_TURNS = 3

encoding = tiktoken.get_encoding("o200k_base")

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)


def count_tokens(text: str) -> int:
    tokens = encoding.encode(text)
    return len(tokens)


def count_message_tokens(messages) -> int:
    total_tokens = 0

    for message in messages:
        total_tokens += count_tokens(message.content)

    return total_tokens


def count_complete_turns(messages):
    human_messages = sum(isinstance(message, HumanMessage) for message in messages)
    ai_messages = sum(isinstance(message, AIMessage) for message in messages)

    return min(human_messages, ai_messages)


def should_summarize(messages):
    return count_complete_turns(messages) >= SUMMARY_TRIGGER


def is_history_over_budget(messages) -> bool:
    total_tokens = count_message_tokens(messages)
    return total_tokens > MAX_HISTORY_TOKENS


def get_system_messages(messages):
    return [message for message in messages if isinstance(message, SystemMessage)]


def split_conversation_history(messages):
    system_messages = get_system_messages(messages)

    conversation_messages = [
        message for message in messages if not isinstance(message, SystemMessage)
    ]

    turns = []
    index = 0
    pending_message = None

    if conversation_messages and isinstance(conversation_messages[-1], HumanMessage):
        pending_message = conversation_messages.pop()

    while index < len(conversation_messages) - 1:
        current_message = conversation_messages[index]
        next_message = conversation_messages[index + 1]

        if isinstance(current_message, HumanMessage) and isinstance(
            next_message, AIMessage
        ):
            turns.append([current_message, next_message])
            index += 2
        else:
            index += 1

    older_turns = turns[:-RECENT_TURNS]
    recent_turns = turns[-RECENT_TURNS:]

    messages_to_summarize = [message for turn in older_turns for message in turn]

    recent_messages = system_messages.copy()

    for turn in recent_turns:
        recent_messages.extend(turn)

    if pending_message:
        recent_messages.append(pending_message)

    return messages_to_summarize, recent_messages


def trim_conversation_history(messages, max_tokens, summary=None):
    system_messages = get_system_messages(messages)

    system_tokens = count_message_tokens(system_messages)
    summary_tokens = count_tokens(summary) if summary else 0

    conversation_messages = [
        message for message in messages if not isinstance(message, SystemMessage)
    ]

    pending_message = None

    if conversation_messages and isinstance(conversation_messages[-1], HumanMessage):
        pending_message = conversation_messages[-1]
        conversation_messages = conversation_messages[:-1]

    pending_tokens = count_message_tokens([pending_message]) if pending_message else 0

    turns = []
    index = 0

    while index < len(conversation_messages) - 1:
        current_message = conversation_messages[index]
        next_message = conversation_messages[index + 1]

        if isinstance(current_message, HumanMessage) and isinstance(
            next_message, AIMessage
        ):
            turns.append([current_message, next_message])
            index += 2
        else:
            index += 1

    retained_turns = []

    remaining_tokens = max_tokens - system_tokens - summary_tokens - pending_tokens

    for turn in reversed(turns):
        turn_tokens = count_message_tokens(turn)

        if turn_tokens > remaining_tokens:
            break

        retained_turns.append(turn)
        remaining_tokens -= turn_tokens

    retained_turns.reverse()

    trimmed_history = system_messages.copy()

    for turn in retained_turns:
        trimmed_history.extend(turn)

    if pending_message:
        trimmed_history.append(pending_message)

    return trimmed_history


def summarize_conversation(messages, existing_summary=None):
    conversation_text = "\n".join(
        f"{type(message).__name__}: {message.content}"
        for message in messages
        if not isinstance(message, SystemMessage)
    )

    previous_summary = existing_summary or "No previous summary available."

    prompt = [
        SystemMessage(
            content=(
                "Update the existing conversation summary using the new messages. "
                "Preserve important facts, user goals, decisions, completed work, "
                "and pending tasks. Keep relevant information from the existing "
                "summary unless the new messages correct it. "
                "Use only information explicitly present in the summary or messages. "
                "Do not answer pending questions or invent information. "
                "Remove greetings and repetition. "
                "Return only the updated summary."
            )
        ),
        HumanMessage(
            content=(
                f"Existing summary:\n{previous_summary}\n\n"
                f"New conversation messages:\n{conversation_text}"
            )
        ),
    ]

    response = llm.invoke(prompt)

    return response.content.strip()


def manage_conversation_context(messages, existing_summary=None):
    if not should_summarize(messages):
        trimmed_messages = trim_conversation_history(
            messages,
            max_tokens=MAX_HISTORY_TOKENS,
            summary=existing_summary,
        )

        return existing_summary, trimmed_messages

    messages_to_summarize, recent_messages = split_conversation_history(messages)

    try:
        updated_summary = summarize_conversation(
            messages_to_summarize,
            existing_summary,
        )

        if not updated_summary:
            print("Summarization returned an empty summary.")

            trimmed_messages = trim_conversation_history(
                messages,
                max_tokens=MAX_HISTORY_TOKENS,
                summary=existing_summary,
            )

            return existing_summary, trimmed_messages

    except Exception as error:
        print(f"Summarization failed: {error}")

        trimmed_messages = trim_conversation_history(
            messages,
            max_tokens=MAX_HISTORY_TOKENS,
            summary=existing_summary,
        )

        return existing_summary, trimmed_messages

    trimmed_messages = trim_conversation_history(
        recent_messages,
        max_tokens=MAX_HISTORY_TOKENS,
        summary=updated_summary,
    )

    return updated_summary, trimmed_messages


if __name__ == "__main__":
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="My name is Alex. I want to learn Python."),
        AIMessage(content="Great! Let's learn Python step by step."),
        HumanMessage(content="What is a variable?"),
        AIMessage(content="A variable stores a value."),
        HumanMessage(content="What is a list?"),
        AIMessage(content="A list stores multiple items."),
        HumanMessage(content="What is a dictionary?"),
        AIMessage(content="A dictionary stores key-value pairs."),
        HumanMessage(content="What is a function?"),
        AIMessage(content="A function is a reusable block of code."),
        HumanMessage(content="What is an exception?"),
        AIMessage(content="An exception is an error during execution."),
        HumanMessage(content="How do I handle exceptions?"),
        AIMessage(content="Use try and except blocks."),
        HumanMessage(content="What is finally used for?"),
        AIMessage(content="Finally executes whether an exception occurs or not."),
        HumanMessage(content="What is a class?"),
        AIMessage(content="A class is a blueprint for creating objects."),
        HumanMessage(content="What is an object?"),
        AIMessage(content="An object is an instance of a class."),
        HumanMessage(content="What is inheritance?"),
        AIMessage(
            content="Inheritance allows a class to reuse another class's functionality."
        ),
        HumanMessage(content="What is encapsulation?"),
        AIMessage(content="Encapsulation bundles data and methods together."),
        HumanMessage(content="What should I learn next?"),
        AIMessage(content="Practice classes and object-oriented programming."),
    ]

    summary, recent_messages = manage_conversation_context(messages)

    print("\nUpdated summary:")
    print(summary)

    print("\nRetained complete turns:", count_complete_turns(recent_messages))
