"""Prompt templates and answer post-processing for the RAG pipeline."""

from __future__ import annotations

NO_CONTEXT_ANSWER = (
    "I don't have enough relevant complaint data to answer that confidently. "
    "Try rephrasing the question or asking about a different product/topic."
)

RAG_PROMPT_TEMPLATE = """You are a financial complaints analyst assistant for CrediTrust.
Use ONLY the complaint excerpts below to answer the question. Do not use outside knowledge.
If the excerpts do not contain enough information to answer, say so explicitly instead of guessing.
Summarize the common themes; do not repeat sentences verbatim from the context.

Context:
{context}

Question:
{question}

Answer (3-4 sentences, grounded only in the context above):
"""


def build_prompt(question: str, context: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


def postprocess_answer(raw_output: str) -> str:
    """Strip the echoed prompt/instructions some causal LMs repeat before the answer."""
    text = raw_output.strip()
    for marker in ("Answer (3-4 sentences, grounded only in the context above):", "Answer:"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text
