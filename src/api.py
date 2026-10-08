"""
FastAPI app providing RAG endpoints. Defines REST API routes for health checks,
document search and answer generation.
"""

from functools import wraps
from typing import Any, Callable, Awaitable
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from src.generate import generate_answer
from src.retrieve import retrieve, preload_retriever
from src.semantic import preload_collection
from src.models import (HealthResponse, SearchRequest, SearchResponse,
                        AnswerRequest, AnswerResponse)


def _http_error(func: Callable[..., Awaitable[Any]]
                ) -> Callable[..., Awaitable[Any]]:
    """
    Decorator to catch exceptions and map them to appropriate HTTP error
    responses.

    Args:
        - func: The async path operation function to decorate.

    Returns:
        The decorated async function.

    Raises:
        HTTPException: Maps caught exceptions to corresponding status codes:
        - 404 (FileNotFoundError)
        - 422 (ValueError)
        - 503 (RuntimeError)
        - 500 (Exception)
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            # Wait for function to be executed
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        # Missing index file (404: Not found)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        # Bad input (422: Unprocessable Entity)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        # Backend unavailable (503)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            print(f"Unhandled error: {type(e).__name__}: {e}",
                  file=sys.stderr)
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper


app = FastAPI()


def _check_bm25_loaded() -> bool:
    """
    Checks if the BM25 retriever index is preloaded and ready.

    Returns:
        True if loaded successfully otherwise False.
    """
    try:
        preload_retriever()
        return True
    except Exception:
        return False


def _check_semantic_loaded() -> bool:
    """
    Checks if the semantic vector collection is preloaded and ready.

    Returns:
        True if loaded successfully otherwise False.
    """
    try:
        preload_collection("data/processed")
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Checks system health and verifies if indexes are ready.

    Returns:
        HealthResponse (check models.py).
    """
    bm25_loaded = _check_bm25_loaded()
    semantic_loaded = _check_semantic_loaded()
    return HealthResponse(
        status="ok",
        index_loaded=bm25_loaded or semantic_loaded,
        bm25_index_loaded=bm25_loaded,
        semantic_index_loaded=semantic_loaded
    )


@app.get("/search", response_model=SearchResponse)
@_http_error
async def api_search(request: SearchRequest) -> SearchResponse:
    """
    Retrieve relevant document sources based on query.

    Args:
        - request: SearchRequest (check models.py).

    Returns:
        SearchResponse (check models.py).
    """
    sources = retrieve(request.query, k=request.k, method=request.method)
    return SearchResponse(query=request.query, k=request.k,
                          method=request.method, sources=sources)


@app.post("/answer", response_model=AnswerResponse)
@_http_error
async def api_answer(request: AnswerRequest) -> AnswerResponse:
    """
    Retrieve sources and generate an answer for the provided query.

    Args:
        - request: AnswerRequest (check models.py).

    Returns:
        AnswerResponse (check models.py).
    """
    sources = retrieve(request.query, k=request.k, method=request.method)
    answer_text = generate_answer(request.query, sources)
    return AnswerResponse(query=request.query, k=request.k,
                          method=request.method, sources=sources,
                          answer=answer_text)


def run_api(host: str = "127.0.0.1", port: int = 8000,
            reload: bool = False) -> None:
    """
    Start the Uvicorn server to host the FastAPI RAG application.

    Args:
        - host: Host address. Defaults to "127.0.0.1".
        - port: Port number to listen on. Defaults to 8000.
        - reload: Enable auto reload. Defaults to False.

    Raises:
        - SystemExit: If the server encounters an OSError
    """
    print(f"Starting RAG API server on http://{host}:{port}")
    print(f"Documentation available at http://{host}:{port}/docs")
    try:
        uvicorn.run("src.api:app", host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as e:
        print(f"Error: could not start server on {host}:{port}: {e},",
              file=sys.stderr)
        sys.exit(1)
