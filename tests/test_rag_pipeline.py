from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from creditrust.config import Settings
from creditrust.rag.llm import MockLLMProvider
from creditrust.rag.pipeline import RAGPipeline
from creditrust.rag.prompts import NO_CONTEXT_ANSWER


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeRetriever:
    def __init__(self, docs: list[FakeDocument]):
        self._docs = docs

    def invoke(self, query: str) -> list[FakeDocument]:
        return self._docs


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_ask_returns_answer_with_sources():
    docs = [
        FakeDocument(
            "Customers report unauthorized charges.", {"complaint_id": "1", "product": "Credit card"}
        ),
        FakeDocument("Fees were charged after payoff.", {"complaint_id": "2", "product": "Personal loan"}),
    ]
    pipeline = RAGPipeline(
        retriever=FakeRetriever(docs),
        llm_provider=MockLLMProvider("Answer: Customers are upset about unexpected fees."),
        settings=_settings(),
    )

    result = pipeline.ask("Why are customers upset?")

    assert result.answer == "Customers are upset about unexpected fees."
    assert result.had_sufficient_context is True
    assert len(result.sources) == 2
    assert result.sources[0].complaint_id == "1"
    assert result.latency_seconds >= 0


def test_ask_guards_against_insufficient_context():
    pipeline = RAGPipeline(
        retriever=FakeRetriever([FakeDocument("", {})]),
        llm_provider=MockLLMProvider("should not be used"),
        settings=_settings(min_context_chars=20),
    )

    result = pipeline.ask("Anything?")

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.had_sufficient_context is False
    assert result.sources == []


def test_ask_rejects_empty_question():
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        llm_provider=MockLLMProvider(),
        settings=_settings(),
    )
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_mock_llm_provider_default_response():
    provider = MockLLMProvider()
    assert "mock" in provider.generate("any prompt").lower()
