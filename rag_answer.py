from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from vector_store import DISTANCE_THRESHOLD, vector_store


class AnswerValidation(BaseModel):
    status: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
    ] = Field(description="Whether the answer is supported by the evidence.")

    supported_claims: list[str] = Field(
        description="Factual claims supported by the document evidence."
    )

    unsupported_claims: list[str] = Field(
        description="Factual claims not supported by the document evidence."
    )

    corrected_answer: str = Field(
        description="An answer containing only supported information."
    )


class QuestionRewrite(BaseModel):
    status: Literal[
        "CLEAR",
        "NEEDS_CLARIFICATION",
    ]

    standalone_question: str = Field(
        description="The standalone question when its meaning is clear."
    )

    clarification_question: str = Field(
        description="A clarification question when the user's intent is ambiguous."
    )


llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)

validator = llm.with_structured_output(
    AnswerValidation,
    method="json_schema",
)

question_rewriter = llm.with_structured_output(
    QuestionRewrite,
    method="json_schema",
)


def rewrite_question(question, conversation_history, summary=None):
    """
    Rewrite follow-up questions and detect ambiguity.
    """

    prompt = [
        SystemMessage(
            content=(
                "You are a question rewriting and ambiguity detection system.\n\n"
                "RULES:\n"
                "1. If the question is already standalone and understandable, "
                "return it unchanged with status CLEAR.\n"
                "2. Use conversation history and summary to resolve references "
                "such as it, they, this, that, and those.\n"
                "3. If the conversation contains exactly one plausible "
                "reference, use it and set status CLEAR.\n"
                "4. If two or more topics are plausible references, "
                "you MUST NOT choose one. Set status NEEDS_CLARIFICATION "
                "and ask the user which topic they mean.\n"
                "5. If no plausible reference exists, set status "
                "NEEDS_CLARIFICATION and ask the user to specify the topic.\n"
                "6. Never resolve ambiguity using outside knowledge.\n"
                "7. Do not invent conversation history or assume the user's intent.\n\n"
                "IMPORTANT EXAMPLES:\n\n"
                "Example A:\n"
                "History: User discussed machine learning only.\n"
                "Question: What are its applications?\n"
                "Result: CLEAR. Rewrite as 'What are the applications "
                "of machine learning?'\n\n"
                "Example B:\n"
                "History: User discussed Python and Java.\n"
                "Question: What are its applications?\n"
                "Result: NEEDS_CLARIFICATION. Ask whether the user means "
                "Python or Java.\n\n"
                "Example C:\n"
                "History: Empty.\n"
                "Question: What are its applications?\n"
                "Result: NEEDS_CLARIFICATION. Ask what topic the user means.\n\n"
                "Previous conversation summary:\n"
                f"{summary or 'No summary available.'}"
            )
        ),
        *conversation_history,
        HumanMessage(content=question),
    ]

    return question_rewriter.invoke(prompt)


def validate_answer(question, context, answer):
    """
    Check whether the generated answer is supported by document evidence.
    Also identify contradictions between different sources.
    """

    prompt = [
        SystemMessage(
            content=(
                "You are a strict answer validation system. "
                "Evaluate ONLY the factual claims in the generated answer. "
                "Never replace the original claims with claims from the evidence. "
                "Break the generated answer into individual factual claims. "
                "Evaluate each claim independently against the document evidence. "
                "IMPORTANT: Treat each document as an independent source. "
                "A claim is SUPPORTED if at least one document directly supports it. "
                "Do not mark a claim unsupported merely because another document "
                "contradicts it. "
                "If different documents support contradictory claims, recognize "
                "this as conflicting evidence. A claim can still be supported "
                "by its own source even when another source disagrees. "
                "Determine the overall status using these exact rules: "
                "SUPPORTED: all factual claims are supported and there are "
                "no unresolved contradictions. "
                "PARTIALLY_SUPPORTED: at least one claim is supported "
                "and at least one claim is unsupported, OR the answer "
                "describes conflicting evidence supported by different sources. "
                "UNSUPPORTED: no factual claims are supported. "
                "The supported_claims field must contain only supported "
                "claims from the original answer. "
                "The unsupported_claims field must contain only unsupported "
                "claims from the original answer. "
                "The corrected_answer must contain only supported claims. "
                "If no claims are supported, return an empty corrected_answer. "
                "Do not use outside knowledge. "
                "Do not invent, rewrite, or add factual claims."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Document evidence:\n{context}\n\n"
                f"Generated answer:\n{answer}"
            )
        ),
    ]

    return validator.invoke(prompt)


def generate_answer(question, conversation_history, summary=None):
    """
    Retrieve relevant document chunks and generate a grounded answer.
    Also return detailed source metadata for attribution.
    """

    rewrite_result = rewrite_question(
        question,
        conversation_history,
        summary,
    )

    if rewrite_result.status == "NEEDS_CLARIFICATION":
        return {
            "answer": rewrite_result.clarification_question,
            "sources": [],
            "standalone_question": None,
            "validation": None,
            "needs_clarification": True,
        }

    standalone_question = rewrite_result.standalone_question

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
            "validation": None,
            "needs_clarification": False,
        }

    context = "\n\n".join(
        (
            f"Source: {document.metadata.get('document_name', 'Unknown source')}\n"
            f"Content:\n{document.page_content}"
        )
        for document, distance in top_chunks
    )

    prompt = [
        SystemMessage(
            content=(
                "Answer the user's question using only the provided document "
                "context. Use conversation history and the previous summary to "
                "understand references and follow-up questions, but do not treat "
                "them as factual evidence. If the document context does not contain "
                "the answer, say you don't know based on the provided document. "
                "Do not invent information.\n\n"
                f"Previous conversation summary:\n"
                f"{summary or 'No summary available.'}"
            )
        ),
        *conversation_history,
        HumanMessage(
            content=(f"Document context:\n{context}\n\nQuestion:\n{question}")
        ),
    ]

    response = llm.invoke(prompt)
    generated_answer = response.content.strip()

    validation = validate_answer(
        standalone_question,
        context,
        generated_answer,
    )

    sources = [
        {
            "chunk_id": document.metadata.get("chunk_id"),
            "document_id": document.metadata.get("document_id"),
            "document_name": document.metadata.get("document_name"),
            "document_version": document.metadata.get("document_version"),
            "file_type": document.metadata.get("file_type"),
            "source": document.metadata.get("source"),
            "page_number": document.metadata.get("page_number"),
            "distance": distance,
            "content": document.page_content,
        }
        for document, distance in top_chunks
    ]

    if validation.status == "UNSUPPORTED" or not validation.corrected_answer.strip():
        final_answer = "I couldn't find a supported answer in the provided document."
    else:
        final_answer = validation.corrected_answer.strip()

    return {
        "answer": final_answer,
        "sources": sources,
        "standalone_question": standalone_question,
        "validation": validation.model_dump(),
        "needs_clarification": False,
    }
