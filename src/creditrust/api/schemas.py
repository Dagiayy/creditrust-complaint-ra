"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Natural-language question.")
    top_k: int | None = Field(
        None, ge=1, le=20, description="Override the number of source chunks retrieved."
    )


class SourceItem(BaseModel):
    complaint_id: str
    product: str
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    latency_seconds: float
    had_sufficient_context: bool


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    llm_provider: str


class ErrorResponse(BaseModel):
    detail: str
