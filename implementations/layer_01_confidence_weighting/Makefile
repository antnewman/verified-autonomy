.PHONY: install test run lint clean

install:
	uv venv .venv
	uv pip install -e ".[dev]"

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=90

run:
	uv run python src/confidence_weighting.py

lint:
	uv run mypy src/
	uv run ruff check src/ tests/

clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache .coverage
