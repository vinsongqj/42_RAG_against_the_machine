"""Local HTTP API for the RAG system."""

from typing import List

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import uvicorn

from src.retriever import retrieve
from src.generator import generate_answer
from src.models import MinimalSource


# ============================================================================
# Pydantic Models for API
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    method: str = "bm25"  # "bm25" | "semantic" | "hybrid"


class SearchResponse(BaseModel):
    query: str
    k: int
    method: str
    sources: List[MinimalSource]


class AnswerRequest(BaseModel):
    query: str
    k: int = 5
    method: str = "bm25"  # "bm25" | "semantic" | "hybrid"


class AnswerResponse(BaseModel):
    query: str
    k: int
    method: str
    sources: List[MinimalSource]
    answer: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    bm25_index_loaded: bool
    semantic_index_loaded: bool


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for codebase Q&A",
    version="1.0.0"
)


def _check_bm25_loaded() -> bool:
    try:
        from src.retriever import _get_retriever
        _get_retriever()
        return True
    except Exception:
        return False


def _check_semantic_loaded() -> bool:
    try:
        from vector_indexer import _get_collection
        _get_collection("data/processed")
        return True
    except Exception:
        return False


def _health_response() -> HealthResponse:
    """Check whether the (cached) BM25 and semantic indexes can be loaded.

    Both health endpoints previously duplicated this try/except and the
    ``/health`` route always returned the stale module-level flag from
    before ``/`` was ever hit, so it reported False even with a valid
    index. Now checked live and reported per-index, since the semantic
    index (Bonus 1) is optional and may not exist even when BM25 does.
    """
    bm25_loaded = _check_bm25_loaded()
    semantic_loaded = _check_semantic_loaded()
    return HealthResponse(
        status="ok",
        index_loaded=bm25_loaded or semantic_loaded,
        bm25_index_loaded=bm25_loaded,
        semantic_index_loaded=semantic_loaded,
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Health check endpoint."""
    return _health_response()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return _health_response()


@app.post("/search", response_model=SearchResponse)
async def api_search(request: SearchRequest) -> SearchResponse:
    """
    Search the index and return top-k sources for a query.
    """
    try:
        sources = retrieve(request.query, k=request.k, method=request.method)
        return SearchResponse(
            query=request.query,
            k=request.k,
            method=request.method,
            sources=sources
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse)
async def api_search_get(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results to return", ge=1, le=50),
    method: str = Query("bm25", description="bm25 | semantic | hybrid"),
) -> SearchResponse:
    """Search the index (GET version)."""
    try:
        sources = retrieve(query, k=k, method=method)
        return SearchResponse(query=query, k=k, method=method, sources=sources)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer", response_model=AnswerResponse)
async def api_answer(request: AnswerRequest) -> AnswerResponse:
    """
    Generate an answer for a query using retrieved context.
    """
    try:
        sources = retrieve(request.query, k=request.k, method=request.method)
        answer_text = generate_answer(request.query, sources)
        return AnswerResponse(
            query=request.query,
            k=request.k,
            method=request.method,
            sources=sources,
            answer=answer_text
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/answer", response_model=AnswerResponse)
async def api_answer_get(
    query: str = Query(..., description="Question to answer"),
    k: int = Query(5, description="Number of sources to retrieve", ge=1, le=50),
    method: str = Query("bm25", description="bm25 | semantic | hybrid"),
) -> AnswerResponse:
    """Generate an answer (GET version)."""
    try:
        sources = retrieve(query, k=k, method=method)
        answer_text = generate_answer(query, sources)
        return AnswerResponse(
            query=query,
            k=k,
            method=method,
            sources=sources,
            answer=answer_text
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the API server."""
    print(f"Starting RAG API server on http://{host}:{port}")
    print(f"Documentation available at http://{host}:{port}/docs")
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_api()