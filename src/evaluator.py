import json
from pathlib import Path
from typing import List
from src.models import AnsweredQuestion, StudentSearchResults, MinimalSource


def compute_recall(student_search_results_path: str,
                   dataset_path: str,
                   k: int = None) -> float:
    """
    Compute recall@k for the student search results against ground truth dataset.
    Returns average recall over all questions.
    """
    # Load student results
    with open(student_search_results_path, "r") as f:
        student_data = json.load(f)
    student_results = StudentSearchResults(**student_data)
    if k is None:
        k = student_results.k

    # Load ground truth dataset
    with open(dataset_path, "r") as f:
        gt_data = json.load(f)
    # The dataset may contain AnsweredQuestion objects
    gt_questions = [AnsweredQuestion(**q) for q in gt_data.get("rag_questions", [])]

    # Build mapping question_id -> ground truth sources
    gt_map = {q.question_id: q.sources for q in gt_questions}

    recalls = []
    for result in student_results.search_results:
        qid = result.question_id
        gt_sources = gt_map.get(qid, [])
        if not gt_sources:
            continue  # skip if no ground truth
        retrieved = result.retrieved_sources[:k]  # top-k

        found = 0
        for gt in gt_sources:
            if _is_covered(gt, retrieved):
                found += 1
        recall = found / len(gt_sources)
        recalls.append(recall)

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def _is_covered(gt: MinimalSource, retrieved: List[MinimalSource]) -> bool:
    """
    Check if any retrieved source covers the ground truth source.
    Coverage: same file and IoU >= 0.05.
    """
    for ret in retrieved:
        if ret.file_path != gt.file_path:
            continue
        if _iou(gt, ret) >= 0.05:
            return True
    return False


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over Union of two character intervals."""
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    inter = max(0, end - start)
    len_a = a.last_character_index - a.first_character_index
    len_b = b.last_character_index - b.first_character_index
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0