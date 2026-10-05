import json
import sys
from functools import wraps                                                    # for pretty-printing dicts as JSON text and loading JSON files
from pathlib import Path                                       # cross-platform file path handling
from typing import Optional                                    # for optional (nullable) type hints
from concurrent.futures import ThreadPoolExecutor, as_completed  # for running retrieval calls concurrently
import fire                                                    # turns plain functions into a CLI automatically
from tqdm import tqdm                                          # progress bars for loops
from src.api import run_api                                    # the function that starts the FastAPI server
from src.evaluate import compute_recall                        # computes recall metric for evaluation
from src.generate import generate_answer                       # calls the LLM to answer a question given sources
from src.index import build_index                              # builds the BM25 (and optionally semantic) index
from src.models import (                                       # pydantic data models used across the CLI
    MinimalAnswer, MinimalSearchResults, RagDataset, StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieve import DEFAULT_DOC_BOOST, preload_retriever, retrieve  # retrieval helpers + the shared default boost, so it can't drift from retrieve.py's own default


def _fail(message: str, code: int = 1) -> None:                # print a one-line error to stderr and exit; SystemExit shows no traceback
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def _cli_guard(func):                                           # decorator: turn any failure in a CLI command into a clean message + exit code
    @wraps(func)                                                # keeps the signature/docstring so python-fire still sees the real parameters
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            _fail("Interrupted.", 130)
        except FileNotFoundError as e:                          # missing index / dataset / results file
            _fail(str(e))
        except json.JSONDecodeError as e:                       # must precede ValueError, which it subclasses
            _fail(f"Invalid JSON input: {e}")
        except ValueError as e:                                 # bad arguments, schema validation failures, unknown method, etc.
            _fail(f"Invalid input: {e}")
        except OSError as e:                                    # permissions, disk full, unwritable output directory
            _fail(f"File system problem: {e}")
        except RuntimeError as e:                               # e.g. Ollama unreachable, embedding/index failures
            _fail(str(e))
        except Exception as e:                                  # last resort: still no traceback
            _fail(f"Unexpected {type(e).__name__}: {e}")
    return wrapper


@_cli_guard
def index(max_chunk_size: int = 2000,                          # CLI command: build the search index from raw docs
          raw_dir: str = "data/raw",                           # folder containing the raw/source files to ingest
          processed_dir: str = "data/processed",               # folder where the built index will be saved
          build_semantic: bool = False) -> None:                # whether to also build the semantic (embedding) index
    build_index(raw_dir,                                       # call the indexing pipeline with these settings
                processed_dir,
                max_chunk_size,
                build_semantic=build_semantic)


@_cli_guard
def search(query: str,                                         # CLI command: run a single search query and print results
           k: int = 5,                                          # number of results to retrieve
           index_dir: str = "data/processed",                   # where the index lives on disk
           method: str = "bm25",                                # retrieval method: bm25 / semantic / hybrid
           doc_boost: float = DEFAULT_DOC_BOOST) -> None:        # score multiplier applied to doc-type sources for conceptual queries
    sources = retrieve(query, k, index_dir, method, doc_boost=doc_boost)  # perform the retrieval
    result = MinimalSearchResults(question_id="",               # wrap results in the response model (no question id for ad-hoc search)
                                  question=query,
                                  retrieved_sources=sources)
    print(json.dumps(result.model_dump(), indent=2))            # pretty-print the result as JSON to stdout


@_cli_guard
def search_dataset(dataset_path: str,                           # CLI command: run retrieval over a whole dataset of questions
                   k: int,                                       # number of results per question
                   save_directory: str,                          # where to write the output JSON file
                   index_dir: str = "data/processed",            # index location
                   num_workers: int = 4,                         # number of threads to use for parallel retrieval
                   use_cache: bool = False,                      # whether to use the on-disk/in-memory query cache
                   method: str = "bm25",                         # retrieval method to use
                   doc_boost: float = DEFAULT_DOC_BOOST) -> None:  # doc-boost factor passed through to retrieval

    with open(dataset_path, "r") as f:                           # open the dataset file containing questions
        data = json.load(f)                                      # parse it as JSON
    questions = RagDataset(**data).rag_questions                 # validate the file's shape and parse each entry as Answered/UnansweredQuestion;
                                                                    # only .question_id/.question are used below, both of which either type has

    if method in ("bm25", "hybrid"):
        preload_retriever(index_dir)                                 # eagerly load/cache the BM25 retriever once, before spinning up threads

    search_results = []                                          # accumulator for per-question results
    with ThreadPoolExecutor(max_workers=num_workers) as executor:  # create a thread pool to parallelize retrieval calls
        future_to_q = {
            executor.submit(retrieve, q.question, k, index_dir, method,   # schedule a retrieve() call for each question
                            use_cache, doc_boost): q
            for q in questions
        }
        for future in tqdm(as_completed(future_to_q),             # iterate over futures as they finish, showing a progress bar
                           total=len(questions),
                           desc="Searching"):
            q = future_to_q[future]                                # map the completed future back to its original question
            try:
                sources = future.result()                          # get the retrieval result (raises if the task errored)
                search_results.append(
                    MinimalSearchResults(question_id=q.question_id,  # build the result record for this question
                                         question=q.question,
                                         retrieved_sources=sources)
                )
            except Exception as e:
                print(f"Error searching question {q.question_id}: {e}")  # log and skip failed questions instead of crashing

    failed = len(questions) - len(search_results)
    if failed:
        print(f"Warning: {failed} of {len(questions)} questions failed and are missing from the output.")

    search_results.sort(key=lambda x: x.question_id)                # sort results by question id for a deterministic output order
    output = StudentSearchResults(search_results=search_results, k=k)  # wrap everything in the top-level output model
    save_path = Path(save_directory) / Path(dataset_path).name       # output file reuses the dataset's file name, in the save directory
    save_path.parent.mkdir(parents=True, exist_ok=True)               # ensure the destination directory exists
    with open(save_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)                  # write the results as pretty JSON
    print(f"Saved StudentSearchResults to {save_path}")               # confirm to the user where the file was saved


@_cli_guard
def answer(query: str, k: int = 5, index_dir: str = "data/processed",  # CLI command: retrieve + generate an answer for one question
           method: str = "bm25", doc_boost: float = DEFAULT_DOC_BOOST) -> None:
    sources = retrieve(query, k, index_dir, method, doc_boost=doc_boost)  # get relevant sources for the query
    answer_text = generate_answer(query, sources)                    # ask the LLM to answer using those sources
    result = MinimalAnswer(question_id="", question=query,           # bundle question, sources, and answer
                           retrieved_sources=sources, answer=answer_text)
    print(json.dumps(result.model_dump(), indent=2))                  # print the full result as JSON


@_cli_guard
def answer_dataset(student_search_results_path: str, save_directory: str,  # CLI command: generate answers for a batch of already-retrieved results
                   max_questions: Optional[int] = None) -> None:            # optionally limit how many questions are processed (e.g. for testing)
    with open(student_search_results_path, "r") as f:                       # load a previously-saved StudentSearchResults file
        data = json.load(f)
    student_data = StudentSearchResults(**data)                             # parse it into the model

    if max_questions is not None:
        student_data.search_results = student_data.search_results[:max_questions]  # truncate to the first N questions
        print(f"Processing only {max_questions} questions (out of {len(student_data.search_results)} total)")

    failed = 0
    answered_results = []                                                    # accumulator for question+sources+answer records
    for res in tqdm(student_data.search_results, desc="Generating answers"):  # loop over each question's search result, with a progress bar
        try:
            ans = generate_answer(res.question, res.retrieved_sources)            # generate an answer using the already-retrieved sources
        except Exception as e:
            failed += 1
            tqdm.write(f"Error answering question {res.question_id}: {type(e).__name__}: {e}")  # skip this one, keep the batch going
            continue
        answered_results.append(
            MinimalAnswer(question_id=res.question_id, question=res.question,  # build the combined record
                          retrieved_sources=res.retrieved_sources, answer=ans)
        )

    if failed:
        print(f"Warning: {failed} question(s) could not be answered and are missing from the output.")

    output = StudentSearchResultsAndAnswer(search_results=answered_results, k=student_data.k)  # wrap in the top-level output model
    save_path = Path(save_directory) / Path(student_search_results_path).name  # reuse the input file's name for the output
    save_path.parent.mkdir(parents=True, exist_ok=True)                         # ensure output directory exists
    with open(save_path, "w") as f:
        json.dump(output.model_dump(), f, indent=2)                            # write results to disk as JSON
    print(f"Saved StudentSearchResultsAndAnswer to {save_path}")                # confirm save location


@_cli_guard
def evaluate(student_search_results_path: str, dataset_path: str, k: Optional[int] = None) -> None:  # CLI command: score retrieval quality
    avg_recall = compute_recall(student_search_results_path, dataset_path, k)   # compute average recall@k across the dataset
    print(f"Recall@{k or 'default'}: {avg_recall:.4f}")                          # print the metric, defaulting the label if k wasn't given


if __name__ == "__main__":                                       # only run the CLI when this file is executed directly (not imported)
    fire.Fire({                                                  # register each function as a CLI subcommand via python-fire
        "index": index,                                          # `python -m src index ...`
        "search": search,                                        # `python -m src search ...`
        "search_dataset": search_dataset,                        # `python -m src search_dataset ...`
        "answer": answer,                                        # `python -m src answer ...`
        "answer_dataset": answer_dataset,                        # `python -m src answer_dataset ...`
        "evaluate": evaluate,                                    # `python -m src evaluate ...`
        "api": _cli_guard(run_api),                                          # `python -m src api ...` starts the web server
    })