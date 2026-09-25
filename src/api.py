from functools import wraps
from typing import Any, Callable, List, TypeVar
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from src.generate import generate_answer
from src.models import MinimalSource
from src.retrieve import retrieve, preload_retriever
from src.semantic_embedding import preload_collection

F = TypeVar("F", bound=Callable[..., Any])


def _as_http_errors(func: F) -> F:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    method: str = "bm25"


class SearchResponse(BaseModel):
    query: str
    k: int
    method: str
    sources: List[MinimalSource]


class AnswerRequest(BaseModel):
    query: str
    k: int = 5
    method: str = "bm25"


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


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for codebase Q&A",
    version="1.0.0",
)


def _check_bm25_loaded() -> bool:
    try:
        preload_retriever()
        return True
    except Exception:
        return False


def _check_semantic_loaded() -> bool:
    try:
        preload_collection("data/processed")
        return True
    except Exception:
        return False


def _health_response() -> HealthResponse:
    bm25_loaded = _check_bm25_loaded()
    semantic_loaded = _check_semantic_loaded()
    return HealthResponse(
        status="ok",
        index_loaded=bm25_loaded or semantic_loaded,
        bm25_index_loaded=bm25_loaded,
        semantic_index_loaded=semantic_loaded,
    )


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return _health_response()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return _health_response()


def _do_search(query: str, k: int, method: str) -> SearchResponse:
    sources = retrieve(query, k=k, method=method)
    return SearchResponse(query=query, k=k, method=method, sources=sources)


def _do_answer(query: str, k: int, method: str) -> AnswerResponse:
    sources = retrieve(query, k=k, method=method)
    answer_text = generate_answer(query, sources)
    return AnswerResponse(query=query, k=k, method=method, sources=sources, answer=answer_text)


@app.post("/search", response_model=SearchResponse)
@_as_http_errors
async def api_search(request: SearchRequest) -> SearchResponse:
    return _do_search(request.query, request.k, request.method)


@app.get("/search", response_model=SearchResponse)
@_as_http_errors
async def api_search_get(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results to return", ge=1, le=50),
    method: str = Query("bm25", description="bm25 | semantic | hybrid"),
) -> SearchResponse:
    return _do_search(query, k, method)


@app.post("/answer", response_model=AnswerResponse)
@_as_http_errors
async def api_answer(request: AnswerRequest) -> AnswerResponse:
    return _do_answer(request.query, request.k, request.method)


@app.get("/answer", response_model=AnswerResponse)
@_as_http_errors
async def api_answer_get(
    query: str = Query(..., description="Question to answer"),
    k: int = Query(5, description="Number of sources to retrieve", ge=1, le=50),
    method: str = Query("bm25", description="bm25 | semantic | hybrid"),
) -> AnswerResponse:
    return _do_answer(query, k, method)


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    print(f"Starting RAG API server on http://{host}:{port}")
    print(f"Documentation available at http://{host}:{port}/docs")
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_api()
