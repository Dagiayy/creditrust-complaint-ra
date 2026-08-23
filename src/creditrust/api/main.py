"""FastAPI service exposing the RAG pipeline for programmatic access.

Endpoints:
    GET  /health          liveness + readiness (is the vector index built?)
    POST /api/v1/query     ask a question, get an answer + cited sources

Auth: if `CREDITRUST_API_KEY` is set, all `/api/v1/*` routes require a
matching `X-API-Key` header. Left unset in local/dev by default.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from creditrust import __version__
from creditrust.config import get_settings
from creditrust.logging_config import get_logger
from creditrust.rag.pipeline import RAGPipeline
from creditrust.rag.retriever import vector_store_exists

from .schemas import ErrorResponse, HealthResponse, QueryRequest, QueryResponse, SourceItem

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="CrediTrust Complaint RAG API",
    description="Retrieval-Augmented Generation over consumer financial complaint narratives.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None

# --- Minimal in-memory rate limiting (per client IP, sliding window) --------
_RATE_LIMIT_REQUESTS = 30
_RATE_LIMIT_WINDOW_SECONDS = 60
_request_log: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = _request_log[client_ip]
        while window and now - window[0] > _RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= _RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again shortly."},
            )
        window.append(now)
    return await call_next(request)


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        if not vector_store_exists(settings.vector_store_dir):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Vector index not built yet. Run "
                    "`python scripts/run_pipeline.py index` and restart the API."
                ),
            )
        _pipeline = RAGPipeline.from_settings(settings)
    return _pipeline


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store_ready=vector_store_exists(settings.vector_store_dir),
        llm_provider=settings.llm_provider,
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
def query(
    payload: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)  # noqa: B008
) -> QueryResponse:
    try:
        result = pipeline.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled error while answering question: %r", payload.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while generating the answer.",
        ) from exc

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceItem(**vars(s)) for s in result.sources],
        latency_seconds=result.latency_seconds,
        had_sufficient_context=result.had_sufficient_context,
    )
