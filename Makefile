.PHONY: install run debug clean lint lint-strict

UV := uv
PYTHON := python
MODULE := src

install:
	@echo "Installing dependencies..."
	$(UV) sync

run:
	@echo "Running script for single query..."
	@echo "Ingesting and indexing the dataset..."
	@$(UV) run $(PYTHON) -m $(MODULE) index --max_chunk_size 2000
	@echo "Searching for single query..."
	@$(UV) run $(PYTHON) -m $(MODULE) search "How does vLLM implement continuous batching?" --k 5
	@echo "Answering for single query..."
	@$(UV) run $(PYTHON) -m $(MODULE) answer "How does vLLM implement continuous batching?" --k 5

run-docs:
	@echo "Running script for dataset..."
	@echo "Ingesting and indexing the dataset..."
	@$(UV) run $(PYTHON) -m $(MODULE) index --max_chunk_size 2000
	@echo "Searching dataset..."
	@$(UV) run $(PYTHON) -m $(MODULE) search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
	--k 10 \
	--save_directory data/output/search_results/UnansweredQuestions \
	--use_cache False
	@echo "Evaluating recall@k..."
	@$(UV) run $(PYTHON) -m $(MODULE) evaluate \
	--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
	--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
	--k 10

run-code:
	@echo "Running script for dataset..."
	@echo "Ingesting and indexing the dataset..."
	@$(UV) run $(PYTHON) -m $(MODULE) index --max_chunk_size 2000
	@echo "Searching dataset..."
	@$(UV) run $(PYTHON) -m $(MODULE) search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
	--k 10 \
	--save_directory data/output/search_results/UnansweredQuestions \
	--use_cache False
	@echo "Evaluating recall@k..."
	@$(UV) run $(PYTHON) -m $(MODULE) evaluate \
	--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
	--dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
	--k 10

answer-docs:
	@echo "Generating answers for 10 questions in the docs dataset..."
	@$(UV) run $(PYTHON) -m $(MODULE) answer_dataset \
	--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions \
    --max_questions 10

answer-code:
	@echo "Generating answers for 10 questions in the code dataset..."
	@$(UV) run $(PYTHON) -m $(MODULE) answer_dataset \
	--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions \
    --max_questions 10

debug:
	@echo "Running in debug mode (pdb)..."
	$(UV) run $(PYTHON) -m pdb -m $(MODULE)

clean:
	@echo "Cleaning temporary files and caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf data/cache data/output data/processed/bm25_index 2>/dev/null || true
	@echo "Cleanup done!"

lint:
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src/
	mypy src/ --strict