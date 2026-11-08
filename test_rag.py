from langchain_core.messages import AIMessage, HumanMessage

from rag_answer import generate_answer, rewrite_question, validate_answer

# --------------------------------------------------
# 1. RAG Answer Generation Test
# --------------------------------------------------

conversation_history = [
    HumanMessage(content="What is machine learning?"),
    AIMessage(content="Machine learning is a branch of artificial intelligence."),
]

question = "What are the applications of deep learning?"

result = generate_answer(question, conversation_history)

print("\n========== RAG ANSWER TEST ==========")

print("\nStandalone question:", result["standalone_question"])
print("\nAnswer:", result["answer"])

print("\nValidation:")
print(result["validation"])

print("\nSources:")

for source in result["sources"]:
    print(f"\nDocument: {source['document_name']}")
    print(f"Chunk ID: {source['chunk_id']}")
    print(f"Document ID: {source['document_id']}")
    print(f"Version: {source['document_version']}")
    print(f"File type: {source['file_type']}")
    print(f"Source: {source['source']}")
    print(f"Page number: {source['page_number']}")
    print(f"Distance: {source['distance']:.4f}")
    print(f"Content: {source['content']}")


# --------------------------------------------------
# 2. Answer Validation Tests
# --------------------------------------------------

validation_question = "What are the applications of deep learning?"

validation_context = (
    "Applications include image recognition, speech recognition, and prediction."
)


def run_validation_test(test_name, answer):
    validation = validate_answer(
        question=validation_question,
        context=validation_context,
        answer=answer,
    )

    print(f"\n========== {test_name} ==========")
    print(validation.model_dump())


run_validation_test(
    "PARTIAL SUPPORT TEST",
    (
        "Deep learning is used for image recognition, "
        "speech recognition, prediction, and autonomous driving."
    ),
)

run_validation_test(
    "UNSUPPORTED TEST",
    (
        "Deep learning is used for weather forecasting, "
        "stock market trading, and autonomous driving."
    ),
)


# --------------------------------------------------
# 3. Question Rewriting Tests
# --------------------------------------------------

rewrite_question_text = "What are its applications?"


def run_rewrite_test(test_name, history):
    result = rewrite_question(
        question=rewrite_question_text,
        conversation_history=history,
    )

    print(f"\n========== {test_name} ==========")
    print("Original question:", rewrite_question_text)
    print("Status:", result.status)
    print("Standalone question:", result.standalone_question)
    print("Clarification:", result.clarification_question)


# Test 1: Empty conversation history

run_rewrite_test(
    "EMPTY HISTORY TEST",
    [],
)


# Test 2: Clear reference

run_rewrite_test(
    "CLEAR REFERENCE TEST",
    [
        HumanMessage(content="What is machine learning?"),
        AIMessage(content="Machine learning is a branch of artificial intelligence."),
    ],
)


# Test 3: Ambiguous reference

ambiguous_history = [
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language."),
    HumanMessage(content="What is Java?"),
    AIMessage(content="Java is also a programming language."),
]

run_rewrite_test(
    "AMBIGUOUS REFERENCE TEST",
    ambiguous_history,
)


# --------------------------------------------------
# 4. End-to-End Ambiguous RAG Test
# --------------------------------------------------

ambiguous_result = generate_answer(
    question=rewrite_question_text,
    conversation_history=ambiguous_history,
)

print("\n========== AMBIGUOUS RAG TEST ==========")
print("Answer:", ambiguous_result["answer"])
print("Needs clarification:", ambiguous_result["needs_clarification"])
print("Standalone question:", ambiguous_result["standalone_question"])
print("Sources:", ambiguous_result["sources"])
print("Validation:", ambiguous_result["validation"])


# --------------------------------------------------
# 5. Conflicting Sources Test
# --------------------------------------------------

conflict_result = generate_answer(
    question=(
        "In conflict_a.txt and conflict_b.txt, "
        "what capital city does each document claim "
        "for the fictional country Testland?"
    ),
    conversation_history=[],
)

print("\n========== CONFLICTING SOURCES TEST ==========")

print("\nStandalone question:", conflict_result["standalone_question"])
print("\nAnswer:", conflict_result["answer"])

print("\nValidation:")
print(conflict_result["validation"])

print("\nSources:")

for source in conflict_result["sources"]:
    print(f"\nDocument: {source['document_name']}")
    print(f"Document ID: {source['document_id']}")
    print(f"Content: {source['content']}")
    print(f"Distance: {source['distance']:.4f}")
