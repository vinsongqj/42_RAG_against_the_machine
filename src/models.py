"""
Data models for the RAG pipeline.

Ingestion and Indexing:
- Chunk

Retrieval and Ranking:
- MinimalSource
- MinimalSearchResults

Generation:
- MinimalAnswer

Evaluation:
- UnansweredQuestion
- AnsweredQuestion
- RagDataset
- StudentSearchResults
- StudentSearchResultsAndAnswer

API:
- SearchRequest
- SearchResponse
- AnswerRequest
- AnswerResponse
- HealthResponse
"""

import uuid
from typing import List, Optional, Union
from pydantic import BaseModel, Field


# ------------------------ INGESTION AND INDEXING -----------------------------


class Chunk(BaseModel):
    """
    A single chunk of a source file created during ingestion/indexing.

    Attributes:
    - file_path: Absolute or relative path to the file.
    - content: The chunk's raw text content.
    - first_character_index: Starting character offset of the chunk within
                             the original file.
    - last_character_index: Ending character offset of the chunk within
                            the original file.
    - score: Optional relevance score, set during retrieval.
    - bm25_text: Optional string combining chunk content and contextual
                 metadata for BM25 indexing.
    """
    file_path: str
    content: str
    first_character_index: int
    last_character_index: int
    score: Optional[float] = None
    bm25_text: Optional[str] = None


# ------------------------ RETRIEVAL AND RANKING ------------------------------


class MinimalSource(BaseModel):
    """
    Lightweight file-location pointer referencing a retrieved chunk without
    including raw content or scores.

    Attributes:
    - file_path: Absolute or relative path to the file.
    - first_character_index: Starting character offset of the chunk within
                             the original file.
    - last_character_index: Ending character offset of the chunk within
                            the original file.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class MinimalSearchResults(BaseModel):
    """
    Retrieval result for a single unanswered question.

    Attributes:
    - question_id: The ID of the question the result belongs to.
    - question: The question text.
    - retrieved_sources: The sources retrieved for the question.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


# ---------------------------- GENERATION -------------------------------------


class MinimalAnswer(MinimalSearchResults):
    """
    Extends MinimalSearchResults with an answer.

    Attributes:
    - answer: The generated answer from the LLM.
    """
    answer: str


# ---------------------------- EVALUATION -------------------------------------


class UnansweredQuestion(BaseModel):
    """
    A question without an answer, assigned a unique ID.

    Attributes:
    - question_id: An auto-generated unique ID.
    - question: The question text.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Extends UnansweredQuestion with the expected sources and answer.

    Attributes:
    - sources: The expected sources for the question.
    - answer: The expected answer text.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    Container for AnsweredQuestions and UnansweredQuestions.

    Attributes:
    - rag_questions: A list of answered questions with expected sources and
                     unanswered questions.
    """
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class StudentSearchResults(BaseModel):
    """
    Batch container holding MinimalSearchResults.

    Attributes:
    - search_results: A list of questions paired with file locations
                      and generated answers.
    - k: The number of top search results requested per question.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Batch container holding MinimalAnswers.

    Attributes:
    - search_results: A list of questions paired with file locations.
    - k: The number of top search results requested per question.
    """
    search_results: List[MinimalAnswer]
    k: int


# -------------------------------- API ----------------------------------------


class SearchRequest(BaseModel):
    """
    Request schema for POST /search.

    Attributes:
    - query: The search query text.
    - k: Number of sources to return.
    - method: Retrieval method ("bm25" / "semantic" / "hybrid").
    """
    query: str
    k: int = 5
    method: str = "bm25"


class SearchResponse(BaseModel):
    """
    Response schema for POST /search.

    Attributes:
    - query: The search query text.
    - k: Number of sources to return.
    - method: Retrieval method used ("bm25" / "semantic" / "hybrid").
    - sources: Retrieved sources.
    """
    query: str
    k: int
    method: str
    sources: List[MinimalSource]


class AnswerRequest(BaseModel):
    """
    Request schema for POST /answer.

    Attributes:
    - query: The question to answer.
    - k: Number of sources to retrieve as context for the answer.
    - method: Retrieval method ("bm25" / "semantic" / "hybrid").
    """
    query: str
    k: int = 5
    method: str = "bm25"


class AnswerResponse(BaseModel):
    """
    Response schema for POST /answer.

    Attributes:
    - query: The question to answer.
    - k: Number of sources to retrieve as context for the answer.
    - method: Retrieval method used ("bm25" / "semantic" / "hybrid").
    - sources: Retrieved sources used to generate answer.
    - answer: The generated answer text.
    """
    query: str
    k: int
    method: str
    sources: List[MinimalSource]
    answer: str


class HealthResponse(BaseModel):
    """
    Response schema for GET /health.

    Attributes:
    - status: Always "ok" while server is running.
    - index_loaded: True if either BM25 or semantic index is loaded.
    - bm25_index_loaded: True if BM25 index is loaded successfully.
    - semantic_index_loaded: True if semantic index is loaded successfully.
    """
    status: str
    index_loaded: bool
    bm25_index_loaded: bool
    semantic_index_loaded: bool
