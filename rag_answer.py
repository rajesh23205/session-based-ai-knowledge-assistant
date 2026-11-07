from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from vector_store import DISTANCE_THRESHOLD, vector_store

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)


def rewrite_question(question, conversation_history, summary=None):
    """
    Convert a follow-up question into a standalone question
    using the previous conversation.
    """

    prompt = [
        SystemMessage(
            content=(
                "You rewrite follow-up questions into standalone questions "
                "using the conversation history. Preserve the user's intent. "
                "If the question is already standalone, return it unchanged. "
                "Return only the rewritten question.\n\n"
                f"Previous conversation summary:\n{summary or 'No summary available.'}"
            )
        ),
        *conversation_history,
        HumanMessage(content=question),
    ]

    response = llm.invoke(prompt)

    return response.content.strip()


def generate_answer(question, conversation_history, summary=None):
    """
    Retrieve relevant document chunks and generate a grounded answer.
    """

    standalone_question = rewrite_question(
        question,
        conversation_history,
        summary,
    )

    results = vector_store.similarity_search_with_score(
        query=standalone_question,
        k=5,
    )

    top_chunks = [
        (document, distance)
        for document, distance in results
        if distance <= DISTANCE_THRESHOLD
    ]

    if not top_chunks:
        return {
            "answer": "I couldn't find an answer in the provided document.",
            "sources": [],
            "standalone_question": standalone_question,
        }

    context = "\n\n".join(document.page_content for document, distance in top_chunks)

    prompt = [
        SystemMessage(
            content=(
                "Answer the user's question using only the provided document "
                "context. Use conversation history and the previous summary to "
                "understand references and follow-up questions, but do not treat "
                "them as factual evidence. If the document context does not contain "
                "the answer, say you don't know based on the provided document. "
                "Do not invent information.\n\n"
                f"Previous conversation summary:\n{summary or 'No summary available.'}"
            )
        ),
        *conversation_history,
        HumanMessage(
            content=(f"Document context:\n{context}\n\nQuestion:\n{question}")
        ),
    ]

    response = llm.invoke(prompt)

    sources = [
        {
            "source": document.metadata["source"],
            "distance": distance,
            "content": document.page_content,
        }
        for document, distance in top_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
        "standalone_question": standalone_question,
    }
