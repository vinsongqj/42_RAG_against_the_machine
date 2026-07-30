.PHONY: install run debug clean lint lint-strict

UV := uv
PYTHON := python
MODULE := src

install:
	@echo "Installing dependencies with uv..."
	$(UV) sync

run:
	@echo "Running the main script..."
	$(UV) run $(PYTHON) -m $(MODULE)

debug:
	@echo "Running in debug mode (pdb)..."
	$(UV) run $(PYTHON) -m pdb -m $(MODULE)

clean:
	@echo "Cleaning temporary files and caches..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf data/cache data/processed/bm25_index 2>/dev/null || true
	@echo "Cleanup done."

lint:
	@echo "Running flake8 and mypy on src/ only..."
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running flake8 and mypy on src/ (strict)..."
	flake8 src/
	mypy src/ --strict