"""End-to-end Retrieval-Augmented Generation pipeline: retrieve -> prompt -> generate.

`RAGPipeline` takes its retriever and LLM provider as constructor arguments
(dependency injection) so it can be exercised in tests with fakes, and so the
Streamlit app, the FastAPI service, and the CLI can all share one
implementation instead of three copy-pasted versions (as in the original
`app.py` / `src/rag_pipeline.py` split).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from creditrust.config import Settings, get_settings
from creditrust.logging_config import get_logger
from creditrust.rag.llm import LLMProvider, get_llm_provider
from creditrust.rag.prompts import NO_CONTEXT_ANSWER, build_prompt, postprocess_answer
from creditrust.rag.retriever import Retriever, get_retriever

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = get_logger(__name__)


@dataclass
class RAGSource:
    complaint_id: str
    product: str
    excerpt: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[RAGSource] = field(default_factory=list)
    latency_seconds: float = 0.0
    had_sufficient_context: bool = True


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        settings: Settings | None = None,
    ):
        self._retriever = retriever
        self._llm = llm_provider
        self._settings = settings or get_settings()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RAGPipeline:
        settings = settings or get_settings()
        return cls(
            retriever=get_retriever(settings.vector_store_dir, settings.top_k),
            llm_provider=get_llm_provider(settings),
            settings=settings,
        )

    def ask(self, question: str) -> RAGAnswer:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        start = time.perf_counter()
        docs: list[Document] = self._retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        if len(context.strip()) < self._settings.min_context_chars:
            logger.warning("Insufficient context (%d chars) for question: %r", len(context), question)
            return RAGAnswer(
                question=question,
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                latency_seconds=time.perf_counter() - start,
                had_sufficient_context=False,
            )

        prompt = build_prompt(question=question, context=context)
        raw_output = self._llm.generate(prompt)
        answer = postprocess_answer(raw_output)

        sources = [
            RAGSource(
                complaint_id=str(doc.metadata.get("complaint_id", "Unknown")),
                product=str(doc.metadata.get("product", "Unknown")),
                excerpt=doc.page_content.strip(),
            )
            for doc in docs
        ]

        latency = time.perf_counter() - start
        logger.info("Answered question in %.2fs using %d source chunks", latency, len(sources))
        return RAGAnswer(question=question, answer=answer, sources=sources, latency_seconds=latency)
