from langchain_core.messages import AIMessage, HumanMessage

from rag_answer import generate_answer

conversation_history = [
    HumanMessage(content="What is machine learning?"),
    AIMessage(content="Machine learning is a branch of artificial intelligence."),
]

question = "What are its applications?"

result = generate_answer(question, conversation_history)

print("\nStandalone question:", result["standalone_question"])
print("\nAnswer:", result["answer"])

print("\nSources:")
for source in result["sources"]:
    print(f"\nSource: {source['source']}")
    print(f"Distance: {source['distance']:.4f}")
    print(f"Content: {source['content']}")
