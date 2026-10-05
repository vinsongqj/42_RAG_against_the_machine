from functools import wraps                                     # preserves function metadata (name, docstring) when wrapping
from typing import Any, Callable, List, TypeVar                   # generic typing helpers for the decorator
import sys
import uvicorn                                                    # ASGI server used to actually run the FastAPI app
from fastapi import FastAPI, HTTPException                        # web framework + error type
from src.generate import generate_answer                          # LLM answer-generation function
from src.retrieve import retrieve, preload_retriever               # retrieval function + public warm-up wrapper, shared with the CLI
from src.semantic import preload_collection               # public warm-up wrapper for the semantic (Chroma) collection
from src.models import HealthResponse, SearchRequest, SearchResponse, AnswerRequest, AnswerResponse, Chunk

F = TypeVar("F", bound=Callable[..., Any])                        # generic type var so the decorator preserves the wrapped function's signature type


def _as_http_errors(func: F) -> F:                                 # decorator: converts internal exceptions into proper HTTP error responses
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise                                                   # already a proper HTTP error: don't re-wrap it as a 500
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))     # missing index/file -> 404 Not Found
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))     # bad input (e.g. invalid method) -> 422 Unprocessable Entity
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))     # backend unavailable (e.g. Ollama down, embedding failure) -> 503
        except Exception as e:
            print(f"Unhandled error in {func.__name__}: {type(e).__name__}: {e}", file=sys.stderr)  # one log line, no traceback
            raise HTTPException(status_code=500, detail=str(e))     # anything else -> 500 Internal Server Error
    return wrapper


app = FastAPI()


def _check_bm25_loaded() -> bool:                                   # try to load the BM25 retriever to see if it's available
    try:
        preload_retriever()
        return True
    except Exception:
        return False


def _check_semantic_loaded() -> bool:                                # try to load the semantic collection to see if it's available
    try:
        preload_collection("data/processed")
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)                   # lets a driving program check the server is up and the index is loaded
async def health() -> HealthResponse:
    bm25_loaded = _check_bm25_loaded()
    semantic_loaded = _check_semantic_loaded()
    return HealthResponse(
        status="ok",
        index_loaded=bm25_loaded or semantic_loaded,
        bm25_index_loaded=bm25_loaded,
        semantic_index_loaded=semantic_loaded,
    )


@app.post("/search", response_model=SearchResponse)                  # query the index
@_as_http_errors
async def api_search(request: SearchRequest) -> SearchResponse:
    sources = retrieve(request.query, k=request.k, method=request.method)
    return SearchResponse(query=request.query, k=request.k, method=request.method, sources=sources)


@app.post("/answer", response_model=AnswerResponse)                  # retrieve + generate an answer
@_as_http_errors
async def api_answer(request: AnswerRequest) -> AnswerResponse:
    sources = retrieve(request.query, k=request.k, method=request.method)
    answer_text = generate_answer(request.query, sources)
    return AnswerResponse(query=request.query, k=request.k, method=request.method,
                          sources=sources, answer=answer_text)


def run_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:  # entry point used by the CLI's `api` command
    print(f"Starting RAG API server on http://{host}:{port}")
    print(f"Documentation available at http://{host}:{port}/docs")
    try:
        uvicorn.run("src.api:app", host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as e:                                            # e.g. port already in use, permission denied on a low port
        print(f"Error: could not start server on {host}:{port}: {e}", file=sys.stderr)
        sys.exit(1)