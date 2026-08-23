from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from creditrust.api import main as api_main
from creditrust.rag.pipeline import RAGAnswer, RAGSource


@dataclass
class _FakePipeline:
    canned: RAGAnswer

    def ask(self, question: str) -> RAGAnswer:
        return self.canned


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point the module-level settings at an empty tmp dir so vector_store_exists() is False by default.
    monkeypatch.setattr(api_main.settings, "vector_store_dir", tmp_path / "no_index_here")
    monkeypatch.setattr(api_main.settings, "api_key", None)
    api_main._pipeline = None
    return TestClient(api_main.app)


def test_health_reports_not_ready_without_index(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["vector_store_ready"] is False
    assert body["status"] == "ok"


def test_query_returns_503_when_index_missing(client):
    response = client.post("/api/v1/query", json={"question": "Why are customers upset?"})
    assert response.status_code == 503


def test_query_returns_answer_when_pipeline_available(client):
    fake_answer = RAGAnswer(
        question="Why are customers upset?",
        answer="Because of unexpected fees.",
        sources=[RAGSource(complaint_id="1", product="Credit card", excerpt="fee complaint")],
        latency_seconds=0.01,
        had_sufficient_context=True,
    )
    api_main.app.dependency_overrides[api_main.get_pipeline] = lambda: _FakePipeline(fake_answer)
    try:
        response = client.post("/api/v1/query", json={"question": "Why are customers upset?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Because of unexpected fees."
        assert body["sources"][0]["complaint_id"] == "1"
    finally:
        api_main.app.dependency_overrides.clear()


def test_query_rejects_empty_question(client):
    # Override the pipeline so this test isolates request-body validation (422)
    # from index-readiness (503) — both would otherwise 4xx/5xx and be ambiguous.
    fake_answer = RAGAnswer("q", "a", [], 0.0, True)
    api_main.app.dependency_overrides[api_main.get_pipeline] = lambda: _FakePipeline(fake_answer)
    try:
        response = client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422
    finally:
        api_main.app.dependency_overrides.clear()


def test_query_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(api_main.settings, "api_key", "secret123")
    fake_answer = RAGAnswer("q", "a", [], 0.0, True)
    api_main.app.dependency_overrides[api_main.get_pipeline] = lambda: _FakePipeline(fake_answer)
    try:
        no_key = client.post("/api/v1/query", json={"question": "x"})
        assert no_key.status_code == 401

        with_key = client.post("/api/v1/query", json={"question": "x"}, headers={"X-API-Key": "secret123"})
        assert with_key.status_code == 200
    finally:
        api_main.app.dependency_overrides.clear()
