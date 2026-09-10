import json
import fire
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models import (
    UnansweredQuestion, MinimalSearchResults, StudentSearchResults,
    MinimalAnswer, StudentSearchResultsAndAnswer
)
from src.indexer import build_index
from src.retriever import retrieve, _get_retriever
from src.generator import generate_answer
from src.evaluator import compute_recall
from src.api import run_api


def index(max_chunk_size: int = 2000,
          raw_dir: str = "data/raw",
          processed_dir: str = "data/processed",
          build_semantic: bool = False) -> None:
    """Ingest raw_dir and build BM25 index under processed_dir.

    Bonus: Semantic embeddings. Pass --build_semantic=true to also build a
    Chroma + MiniLM vector index next to the BM25 index. Defaults to False
    so the command's behaviour and timing are unchanged unless opted in.
    """
    build_index(raw_dir, processed_dir, max_chunk_size, build_semantic=build_semantic)


def search(query: str, k: int = 5, index_dir: str = "data/processed",
           method: str = "bm25") -> None:
    """Single‑query search: print top-k sources as JSON.

    Bonus: Hybrid retrieval. ``method`` is "bm25" (default, lexical only),
    "semantic" (vector only), or "hybrid" (Reciprocal Rank Fusion of both).
    "semantic"/"hybrid" require ``index --build_semantic true`` to have
    been run first.
    """
    sources = retrieve(query, k, index_dir, method)
    result = MinimalSearchResults(
        question_id="",
        question=query,
        retrieved_sources=sources
    )
    print(json.dumps(result.model_dump(), indent=2))


def search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    index_dir: str = "data/processed",
    num_workers: int = 4,
    use_cache: bool = False,
    method: str = "bm25",
) -> None:
    """Batch search over a dataset with parallel processing.

    Bonus: Hybrid retrieval. ``method`` is "bm25" (default), "semantic",
    or "hybrid" -- see `search` for details.

    ``use_cache`` is passed straight through to ``retrieve()`` rather than
    monkeypatching ``query_cache.get``/``set`` in place: that approach
    mutated shared global state (unsafe under this function's own thread
    pool) and reassigned methods to incompatible types, which mypy
    correctly flags as an error.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)
    questions = [UnansweredQuestion(**q) for q in data.get("rag_questions", data)]

    # Pre‑load the index once so every worker thread shares it
    _ = _get_retriever(index_dir)

    search_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_q = {
            executor.submit(retrieve, q.question, k, index_dir, method, use_cache): q
            for q in questions
        }
        for future in tqdm(as_completed(future_to_q), total=len(questions), desc="Searching"):
            q = future_to_q[future]
            try:
                sources = future.result()
                search_results.append(
                    MinimalSearchResults(
                        question_id=q.question_id,
                        question=q.question,
                        retrieved_sources=sources
                    )
                )
            except Exception as e:
                print(f"Error searching question {q.question_id}: {e}")

    search_results.sort(key=lambda x: x.question_id)
    output = StudentSearchResults(search_results=search_results, k=k)
    save_path = Path(save_directory) / Path(dataset_path).name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)
    print(f"Saved StudentSearchResults to {save_path}")


def answer(query: str, k: int = 5, index_dir: str = "data/processed",
           method: str = "bm25") -> None:
    """Single‑query answer generation.

    Bonus: Hybrid retrieval. ``method`` is "bm25" (default), "semantic",
    or "hybrid" -- see `search` for details.
    """
    sources = retrieve(query, k, index_dir, method)
    answer_text = generate_answer(query, sources)
    result = MinimalAnswer(
        question_id="",
        question=query,
        retrieved_sources=sources,
        answer=answer_text
    )
    print(json.dumps(result.model_dump(), indent=2))


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
    max_questions: Optional[int] = None,
) -> None:
    """Generate answers from existing StudentSearchResults, optionally limiting."""
    with open(student_search_results_path, "r") as f:
        data = json.load(f)
    student_data = StudentSearchResults(**data)

    if max_questions is not None:
        student_data.search_results = student_data.search_results[:max_questions]
        print(f"Processing only {max_questions} questions (out of {len(student_data.search_results)} total)")

    answered_results = []
    for res in tqdm(student_data.search_results, desc="Generating answers"):
        ans = generate_answer(res.question, res.retrieved_sources)
        answered_results.append(
            MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=ans
            )
        )

    output = StudentSearchResultsAndAnswer(
        search_results=answered_results,
        k=student_data.k
    )
    save_path = Path(save_directory) / Path(student_search_results_path).name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)
    print(f"Saved StudentSearchResultsAndAnswer to {save_path}")


def evaluate(student_search_results_path: str,
             dataset_path: str,
             k: Optional[int] = None) -> None:
    """Compute recall@k against ground truth dataset."""
    avg_recall = compute_recall(student_search_results_path, dataset_path, k)
    print(f"Recall@{k or 'default'}: {avg_recall:.4f}")


if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate,
        "api": run_api
    })