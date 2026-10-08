"""
Command Line Interface for the RAG system.
Provides CLI commands powered by Python Fire for building search
indexes, running queries, processing datasets in bulk, generating answers,
evaluating recall@k, and starting the REST API server.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import json
from pathlib import Path
import sys
from typing import Optional

import fire
from tqdm import tqdm

from src.api import run_api
from src.evaluate import compute_recall
from src.generate import generate_answer
from src.index import build_index
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieve import DEFAULT_DOC_BOOST, preload_retriever, retrieve


def _fail(message: str, code: int = 1) -> None:
    """
    Print an error message to stderr and exit the application cleanly.

    Args:
        - message: Error details to display to the user.
        - code: System exit status code. Defaults to 1.
    """
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def _cli_guard(func):
    """
    Decorator that catches runtime exceptions and outputs formatted
    CLI errors.

    Args:
        - func: The CLI target function to decorate.

    Returns:
        The wrapped function returning clean error exit codes instead
        of raising tracebacks.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            _fail("Interrupted.", 130)
        except FileNotFoundError as e:
            _fail(str(e))
        except json.JSONDecodeError as e:
            _fail(f"Invalid JSON input: {e}")
        except ValueError as e:
            _fail(f"Invalid input: {e}")
        except OSError as e:
            _fail(f"File system problem: {e}")
        except RuntimeError as e:
            _fail(str(e))
        except Exception as e:
            _fail(f"Unexpected {type(e).__name__}: {e}")

    return wrapper


@_cli_guard
def index(max_chunk_size: int = 2000, raw_dir: str = "data/raw",
          processed_dir: str = "data/processed",
          build_semantic: bool = False) -> None:
    """
    Build document search indexes from raw source files.

    Args:
        - max_chunk_size: Maximum character length for document chunks.
                          Defaults to 2000.
        - raw_dir: Path to directory containing raw files.
                   Defaults to "data/raw".
        - processed_dir: Destination path for built indexes.
                         Defaults to "data/processed".
        - build_semantic: Whether to construct vector embeddings for
                          semantic search. Defaults to False.
    """
    build_index(
        raw_dir, processed_dir, max_chunk_size, build_semantic=build_semantic
    )


@_cli_guard
def search(query: str, k: int = 5, index_dir: str = "data/processed",
           method: str = "bm25", doc_boost: float = DEFAULT_DOC_BOOST) -> None:
    """Run a search query and output JSON results to stdout.

    Args:
        - query: Search string to query against the index.
        - k: Maximum number of sources to retrieve. Defaults to 5.
        - index_dir: Directory containing prebuilt search indexes.
                     Defaults to "data/processed".
        - method: Retrieval algorithm ("bm25", "semantic", or "hybrid").
                  Defaults to "bm25".
        - doc_boost: Score multiplier applied to document sources.
                     Defaults to DEFAULT_DOC_BOOST.
    """
    sources = retrieve(query, k, index_dir, method, doc_boost=doc_boost)
    result = MinimalSearchResults(
        question_id="", question=query, retrieved_sources=sources
    )
    print(json.dumps(result.model_dump(), indent=2))


@_cli_guard
def search_dataset(dataset_path: str, k: int, save_directory: str,
                   index_dir: str = "data/processed", num_workers: int = 4,
                   use_cache: bool = False, method: str = "bm25",
                   doc_boost: float = DEFAULT_DOC_BOOST) -> None:
    """Run parallel search queries across a multi-question evaluation dataset.

    Args:
        - dataset_path: Path to input dataset JSON file containing questions.
        - k: Number of retrieved sources per question.
        - save_directory: Target directory where output JSON will be written.
        - index_dir: Path to directory holding built indexes.
                     Defaults to "data/processed".
        - num_workers: Number of concurrent threads for retrieval execution.
                       Defaults to 4.
        - use_cache: Whether to store and reuse query results from cache.
                     Defaults to False.
        - method: Retrieval method ("bm25", "semantic", or "hybrid").
                  Defaults to "bm25".
        - doc_boost: Score multiplier applied to document sources.
                     Defaults to DEFAULT_DOC_BOOST.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)
    questions = RagDataset(**data).rag_questions

    if method in ("bm25", "hybrid"):
        preload_retriever(index_dir)

    search_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_q = {
            executor.submit(retrieve, q.question, k, index_dir, method,
                            use_cache, doc_boost): q
            for q in questions
        }
        for future in tqdm(
            as_completed(future_to_q),
            total=len(questions),
            desc="Searching",
        ):
            q = future_to_q[future]
            try:
                sources = future.result()
                search_results.append(
                    MinimalSearchResults(
                        question_id=q.question_id,
                        question=q.question,
                        retrieved_sources=sources,
                    )
                )
            except Exception as e:
                print(f"Error searching question {q.question_id}: {e}")

    failed = len(questions) - len(search_results)
    if failed:
        print(f"Warning: {failed} of {len(questions)} questions failed and are"
              " missing from the output.")

    search_results.sort(key=lambda x: x.question_id)
    output = StudentSearchResults(search_results=search_results, k=k)
    save_path = Path(save_directory) / Path(dataset_path).name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)
    print(f"Saved StudentSearchResults to {save_path}")


@_cli_guard
def answer(query: str, k: int = 5, index_dir: str = "data/processed",
           method: str = "bm25", doc_boost: float = DEFAULT_DOC_BOOST) -> None:
    """
    Retrieve relevant documents and generate an answer for a single query.

    Args:
        - query: Question text to retrieve context and answer for.
        - k: Maximum number of sources to retrieve. Defaults to 5.
        - index_dir: Path to directory holding built indexes.
                     Defaults to "data/processed".
        - method: Retrieval method ("bm25", "semantic", or "hybrid").
                  Defaults to "bm25".
        - doc_boost: Score multiplier applied to document sources.
                     Defaults to DEFAULT_DOC_BOOST.
    """
    sources = retrieve(query, k, index_dir, method, doc_boost=doc_boost)
    answer_text = generate_answer(query, sources)
    result = MinimalAnswer(
        question_id="",
        question=query,
        retrieved_sources=sources,
        answer=answer_text,
    )
    print(json.dumps(result.model_dump(), indent=2))


@_cli_guard
def answer_dataset(student_search_results_path: str, save_directory: str,
                   max_questions: Optional[int] = None) -> None:
    """
    Generate LLM answers for a batch of pre-retrieved dataset search results.

    Args:
        - student_search_results_path: Path to JSON file containing
                                       previous search results.
        - save_directory: Destination directory for saving generated answers.
        - max_questions: Optional limit on the number of questions to process.
                         Defaults to None (processes all).
    """
    with open(student_search_results_path, "r") as f:
        data = json.load(f)
    student_data = StudentSearchResults(**data)

    if max_questions is not None:
        student_data.search_results = student_data.search_results[
            :max_questions
        ]
        print(f"Processing only {max_questions} questions "
              f"(out of {len(student_data.search_results)} total)")

    failed = 0
    answered_results = []
    for res in tqdm(student_data.search_results, desc="Generating answers"):
        try:
            ans = generate_answer(res.question, res.retrieved_sources)
        except Exception as e:
            failed += 1
            tqdm.write(
                f"Error answering question {res.question_id}: "
                f"{type(e).__name__}: {e}"
            )
            continue
        answered_results.append(
            MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=ans,
            )
        )

    if failed:
        print(f"Warning: {failed} question(s) could not be answered and "
              "are missing from the output.")

    output = StudentSearchResultsAndAnswer(
        search_results=answered_results, k=student_data.k
    )
    save_path = Path(save_directory) / Path(student_search_results_path).name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)
    print(f"Saved StudentSearchResultsAndAnswer to {save_path}")


@_cli_guard
def evaluate(student_search_results_path: str, dataset_path: str,
             k: Optional[int] = None) -> None:
    """
    Evaluate and compute the average Recall@k for dataset search results.

    Args:
        - student_search_results_path: Path to search results file to evaluate.
        - dataset_path: Path to ground-truth dataset JSON file.
        - k: Optional evaluation cut-off point k. Defaults to None.
    """
    avg_recall = compute_recall(student_search_results_path, dataset_path, k)
    print(f"Recall@{k or 'default'}: {avg_recall:.4f}")


if __name__ == "__main__":
    fire.Fire(
        {
            "index": index,
            "search": search,
            "search_dataset": search_dataset,
            "answer": answer,
            "answer_dataset": answer_dataset,
            "evaluate": evaluate,
            "api": _cli_guard(run_api),
        }
    )
