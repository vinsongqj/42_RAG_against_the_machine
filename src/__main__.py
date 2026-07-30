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


def index(max_chunk_size: int = 2000,
         raw_dir: str = "data/raw",
         processed_dir: str = "data/processed"):
    """Ingest raw_dir and build BM25 index under processed_dir."""
    build_index(raw_dir, processed_dir, max_chunk_size)


def search(query: str, k: int = 5, index_dir: str = "data/processed"):
    """Single‑query search: print top-k sources as JSON."""
    sources = retrieve(query, k, index_dir)
    result = MinimalSearchResults(
        question_id="",
        question=query,
        retrieved_sources=sources
    )
    print(json.dumps(result.model_dump(), indent=2))


def search_dataset(dataset_path: str,
                   k: int,
                   save_directory: str,
                   index_dir: str = "data/processed",
                   num_workers: int = 4,
                   use_cache: bool = False):  # new flag
    """Batch search over a dataset with parallel processing."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    questions = [UnansweredQuestion(**q) for q in data.get("rag_questions", data)]

    # Pre‑load the index once
    _ = _get_retriever(index_dir)

    # Temporarily disable cache if not using
    if not use_cache:
        from src.retriever import query_cache
        original_get = query_cache.get
        original_set = query_cache.set
        query_cache.get = lambda key: None
        query_cache.set = lambda key, value: None

    try:
        search_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_q = {
                executor.submit(retrieve, q.question, k, index_dir): q
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
    finally:
        # Restore cache if disabled
        if not use_cache:
            query_cache.get = original_get
            query_cache.set = original_set

    search_results.sort(key=lambda x: x.question_id)
    output = StudentSearchResults(search_results=search_results, k=k)
    save_path = Path(save_directory) / Path(dataset_path).name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)
    print(f"Saved StudentSearchResults to {save_path}")


def answer(query: str, k: int = 5, index_dir: str = "data/processed"):
    """Single‑query answer generation."""
    sources = retrieve(query, k, index_dir)
    answer_text = generate_answer(query, sources)
    result = MinimalAnswer(
        question_id="",
        question=query,
        retrieved_sources=sources,
        answer=answer_text
    )
    print(json.dumps(result.model_dump(), indent=2))


def answer_dataset(student_search_results_path: str,
                   save_directory: str,
                   max_questions: Optional[int] = None):
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
             k: Optional[int] = None):
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
    })