*This project has been created as part of the 42 curriculum by vgoh*

# RAG against the machine

**Table of contents**
- [RAG against the machine](#rag-against-the-machine)
  - [Description](#description)
  - [Instructions](#instructions)
    - [Quick commands](#quick-commands)
    - [Manual usage](#manual-usage)
    - [Example usage](#example-usage)
  - [System architecture](#system-architecture)
  - [Chunking strategy](#chunking-strategy)
  - [Retrieval method](#retrieval-method)
  - [Performance analysis](#performance-analysis)
  - [Design decisions](#design-decisions)
  - [Challenges faced](#challenges-faced)
  - [Resources](#resources)
    - [Disclosure of AI Usage](#disclosure-of-ai-usage)


## Description

**RAG against the machine** is a Retrieval-Augmented Generation (RAG) project pipeline that ingests and indexes a provided codebase, retrieves the most relevant snippets for a question, hands them to a small local model (Qwen3-0.6B) to generate a grounded answer, and measures retrieval quality with recall@k.

[↑ Back to Table of Contents](#rag-against-the-machine)

## Instructions

### Quick commands
A Makefile has been created for convenience. After cloning the repository, you may run the following commands:
```bash
make install       # Installs the dependencies required
make run           # Runs the pipeline for a single default query 
make run-docs      # Runs the pipeline for the default docs dataset
make run-code      # Runs the pipeline for the default code dataset
make answer-docs   # Generates the answer for the docs dataset
make answer-code   # Generates the answer for the code dataset
make debug         # Debugs the program with python debugger
make lint          # Runs mypy and flake8 linting tests
make lint-strict   # Runs mypy with the --strict flag and flake8
make clean         # Removes all build files
```
To alter the defaults, please proceed to the Makefile and edit the necessary file paths, k values or queries. The answer commands have been kept separate from the pipeline due to performance incompatability issues with lower end devices, so the run commands are mainly to evaluate recall@k rankings.

[↑ Back to Table of Contents](#rag-against-the-machine)

### Manual usage
If you would like to run the program manually, you may input the following commands in the terminal:

1. Install the dependencies using 
   ```bash
   uv sync
   ```
2. Ingest the data and build the index using 
    ```bash
    uv run python -m src index -max_chunk_size <int>
    ```
3. You may choose to either run a search
    * For a single query:
        ```bash
        uv run python -m src answer <query> -k <int>
        ```
    * Or across an entire dataset: 
        ```bash
        uv run python -m src search dataset -dataset_path <path> -save_directory <directory>
        ```
4. Then you can evaluate the recall@k for the dataset search:
   ```bash
   uv run python -m src evaluate –student_search_results_path <path> –dataset_path <path>
   ```
5. Otherwise, you may generate the answers
    * For the single query:
        ```bash
        uv run python -m src search <query> -k <int>
        ```
    * Or across the entire dataset: 
        ```bash
        uv run python -m src answer_dataset –student_search_results_path <path> –save_directory <directory>
        ```

[↑ Back to Table of Contents](#rag-against-the-machine)

### Example usage

1. First we need build the index with an example max chunk size of 2000:
   ```bash
   uv run python -m src index --max_chunk_size 2000
   ```

2. Then we search the dataset:
    ```bash
    uv run python -m src search_dataset \
     --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
     --k 10 \
     --save_directory data/output/search_results/UnansweredQuestions \
     --use_cache False
    ```
    This gives us a JSON file in `data/output/search_results/UnansweredQuestions` (truncated for readability):
    ```json
    {
      "search_results": [
        {
          "question_id": "q1",
          "question": "How to configure OpenAI compatible server?",
          "retrieved_sources": [
            {
              "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
              "first_character_index": 9867,
              "last_character_index": 10100
            },
            {
              "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
              "first_character_index": 267,
              "last_character_index": 400
            },
            {
              "file_path": "data/raw/vllm-0.10.1/docs/serving/deploying_with_docker.md",
              "first_character_index": 534,
              "last_character_index": 723
            }
          ]
        },
        {
          "question_id": "q2",
          "question": "How does vLLM handle continuous batching?",
          "retrieved_sources": [
            {
              "file_path": "data/raw/vllm-0.10.1/vllm/core/scheduler.py",
              "first_character_index": 1245,
              "last_character_index": 1567
            },
            {
              "file_path": "data/raw/vllm-0.10.1/docs/design/continuous_batching.md",
              "first_character_index": 0,
              "last_character_index": 234
            }
          ]
        }
      ],
      "k": 10
    }
    ```

3. Once we have the search results, we can evaluate our recall@k ranking:
   ```bash
   uv run python -m src evaluate \
   --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
   --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
   --k 10
   ```
   With a k of 10, the recall@k ranking for this search result should be `Recall@10: 0.9100`.

4. We can generate the answers using:
    ```bash
    uv run python -m src answer_dataset \
     --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
     --save_directory data/output/search_results_and_answer/UnansweredQuestions \
     --max_questions 10
    ```
    This creates a JSON file with the search results and the answer for 10 questions in `data/output/search_results_and_answer/UnansweredQuestions` (truncated for readability):
    ```json
    {
      "search_results": [
        {
          "question_id": "q1",
          "question": "How to configure OpenAI compatible server?",
          "retrieved_sources": [
            {
              "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
              "first_character_index": 9867,
              "last_character_index": 10100
            }
          ],
          "answer": "To configure the OpenAI compatible server in vLLM, you need to start the server with the --api-server flag and specify the model path. You can also configure parameters like --port, --host, and --max-num-batched-tokens."
        }
      ],
      "k": 10
    }
    ```
    Do be warned that this will take a long time to generate depending on your device specs.

[↑ Back to Table of Contents](#rag-against-the-machine)

## System architecture

[↑ Back to Table of Contents](#rag-against-the-machine)

## Chunking strategy

[↑ Back to Table of Contents](#rag-against-the-machine)

## Retrieval method

[↑ Back to Table of Contents](#rag-against-the-machine)

## Performance analysis

[↑ Back to Table of Contents](#rag-against-the-machine)

## Design decisions

[↑ Back to Table of Contents](#rag-against-the-machine)

## Challenges faced

[↑ Back to Table of Contents](#rag-against-the-machine)

## Resources

[↑ Back to Table of Contents](#rag-against-the-machine)

### Disclosure of AI Usage

DeepSeek was used to answer some more in-depth questions I had about chunking and retrieval, as well as error handling and finding edge cases in my pipeline, while Gemini was used for asking basic questions about topics such as tqdm, Fire CLI and how to include a table of contents in my readme.

[↑ Back to Table of Contents](#rag-against-the-machine)