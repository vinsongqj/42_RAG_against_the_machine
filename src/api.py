"""Local HTTP API for the RAG system."""

import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
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


class SearchResponse(BaseModel):
    query: str
    k: int
    sources: List[MinimalSource]


class AnswerRequest(BaseModel):
    query: str
    k: int = 5


class AnswerResponse(BaseModel):
    query: str
    k: int
    sources: List[MinimalSource]
    answer: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for codebase Q&A",
    version="1.0.0"
)

# Global state
_index_loaded = False


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    global _index_loaded
    try:
        # Try to load index (cached)
        from src.retriever import _get_retriever
        _get_retriever()
        _index_loaded = True
    except Exception:
        _index_loaded = False
    return HealthResponse(status="ok", index_loaded=_index_loaded)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global _index_loaded
    return HealthResponse(status="ok", index_loaded=_index_loaded)


@app.post("/search", response_model=SearchResponse)
async def api_search(request: SearchRequest):
    """
    Search the index and return top-k sources for a query.
    """
    try:
        sources = retrieve(request.query, k=request.k)
        return SearchResponse(
            query=request.query,
            k=request.k,
            sources=sources
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def api_search_get(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results to return", ge=1, le=50)
):
    """Search the index (GET version)."""
    try:
        sources = retrieve(query, k=k)
        return SearchResponse(query=query, k=k, sources=sources)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer", response_model=AnswerResponse)
async def api_answer(request: AnswerRequest):
    """
    Generate an answer for a query using retrieved context.
    """
    try:
        sources = retrieve(request.query, k=request.k)
        answer_text = generate_answer(request.query, sources)
        return AnswerResponse(
            query=request.query,
            k=request.k,
            sources=sources,
            answer=answer_text
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/answer")
async def api_answer_get(
    query: str = Query(..., description="Question to answer"),
    k: int = Query(5, description="Number of sources to retrieve", ge=1, le=50)
):
    """Generate an answer (GET version)."""
    try:
        sources = retrieve(query, k=k)
        answer_text = generate_answer(query, sources)
        return AnswerResponse(
            query=query,
            k=k,
            sources=sources,
            answer=answer_text
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CLI Entry Point
# ============================================================================

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    print(f"🚀 Starting RAG API server on http://{host}:{port}")
    print(f"📚 Documentation available at http://{host}:{port}/docs")
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_api()