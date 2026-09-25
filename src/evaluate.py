import json
from typing import List, Optional
from src.models import AnsweredQuestion, RagDataset, StudentSearchResults, MinimalSource


def compute_recall(
    student_search_results_path: str,
    dataset_path: str,
    k: Optional[int] = None,
) -> float:
    with open(student_search_results_path, "r") as f:
        student_data = json.load(f)
    student_results = StudentSearchResults(**student_data)
    if k is None:
        k = student_results.k

    with open(dataset_path, "r") as f:
        gt_data = json.load(f)
    gt_questions = RagDataset(**gt_data).rag_questions
    for q in gt_questions:
        if not isinstance(q, AnsweredQuestion):
            raise ValueError(
                f"Ground-truth question {q.question_id!r} in {dataset_path} "
                "is missing its 'sources'/'answer' fields."
            )
    gt_map = {q.question_id: q.sources for q in gt_questions}

    recalls = []
    for result in student_results.search_results:
        gt_sources = gt_map.get(result.question_id, [])
        if not gt_sources:
            continue
        retrieved = result.retrieved_sources[:k]
        found = sum(1 for gt in gt_sources if _is_covered(gt, retrieved))
        recalls.append(found / len(gt_sources))

    return sum(recalls) / len(recalls) if recalls else 0.0


def _is_covered(gt: MinimalSource, retrieved: List[MinimalSource]) -> bool:
    return any(
        ret.file_path == gt.file_path and _iou(gt, ret) >= 0.05
        for ret in retrieved
    )


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    inter = max(0, end - start)
    len_a = a.last_character_index - a.first_character_index
    len_b = b.last_character_index - b.first_character_index
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0
