"""
Computes recall@k score to evaluate retrieval.
"""

import json
from typing import Any, List, Optional
from pydantic import ValidationError
from src.models import (AnsweredQuestion, RagDataset,
                        StudentSearchResults, MinimalSource)


def _load_json(path: str, what: str) -> Any:
    """
    Reads a JSON file, converting failures into clear messages.

    Args:
        - path: The JSON file path.
        - what: A description for the content expected in the JSON file.

    Returns:
        The deserialized JSON content.

    Raises:
        - FileNotFoundError: If the file can't be found.
        - ValueError: If the file contains invalid syntax or decoding errors.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{what} file not found: {path}") from None
    except ValueError as e:
        raise ValueError(f"{what} file {path} is not valid JSON") from e


def compute_recall(
        student_search_results_path: str, dataset_path: str,
        k: Optional[int] = None) -> float:
    """
    Computes mean retrieval recall across all evaluated questions in the
    dataset. Evaluates retrieval predictions against ground truth dataset by
    comparing character overlap between expected sources and top-k retrieved
    sources.

    Args:
        - student_search_results_path: File path to the JSON containing student
                                       search results.
        - dataset_path: File path to the ground truth dataset JSON.

    Returns:
        The average recall score across all questions that have matching ground
        truth sources (0.0 - 1.0). Returns 0.0 if no question IDs match.

    Raises:
        - ValueError: If k is negative, if either input file fails Pydantic
                      validation or if ground truth questions lack expected
                      sources/answer fields.
    """
    student_data = _load_json(student_search_results_path, "Search results")
    try:
        student_results = StudentSearchResults.model_validate(student_data)
    except ValidationError as e:
        raise ValueError("Invalid search results "
                         f" file {student_search_results_path}: {e}") from e

    if k is None:
        k = student_results.k
    if k <= 0:
        raise ValueError(f"k must be a positive int, got {k}")

    target_data = _load_json(dataset_path, "Dataset")
    try:
        target_questions = RagDataset.model_validate(target_data).rag_questions
    except ValidationError as e:
        raise ValueError(f"Dataset file {dataset_path} has unexpected"
                         " structure") from e

    for question in target_questions:
        if not isinstance(question, AnsweredQuestion):
            raise ValueError(f"Target question in {question.question_id!r} in "
                             f"{dataset_path} is missing its sources/answer "
                             "fields")
    target_map = {question.question_id: question.sources
                  for question in target_questions}
    recalls = []
    for result in student_results.search_results:
        target_sources = target_map.get(result.question_id, [])
        if not target_sources:
            continue
        retrieved = result.retrieved_sources[:k]
        found = sum(1 for source in target_sources
                    if _is_found(source, retrieved))
        recalls.append(found / len(target_sources))

    if not recalls:
        print("No question ids matched between results and dataset."
              " Recall = 0")
    return sum(recalls) / len(recalls) if recalls else 0.0


def _is_found(target: MinimalSource, retrieved: List[MinimalSource]) -> bool:
    """
    Determines if a target source snippet is covered by retrieved sources.
    Considered covered if any retrieved snippet shares file path and has
    an Intersection-over-Union of at least 5% of character indices.

    Args:
        - target: The ground truth source snippet to check.
        - retrieved: A list of retrieved sources to search through.

    Returns:
        True if at least one retrieved source matches the target snippet else
        False.
    """
    return any(
        r.file_path == target.file_path and _iou(target, r) >= 0.05
        for r in retrieved
    )


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    """
    Calculates Intersection-over-Union of character spans between two sources.

    Args:
        - a: The first source snippet containing character boundaries.
        - b: The second source snippet containing character boundaries.

    Returns:
        The character span IoU score between 0.0 and 1.0. Returns 0.0 if the
        ranges do not intersect or if the combined union length is 0.
    """
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    inter = max(0, end - start)  # 0 if no intersection

    len_a = a.last_character_index - a.first_character_index
    len_b = b.last_character_index - b.first_character_index
    union = len_a + len_b - inter

    return inter / union if union > 0 else 0.0
