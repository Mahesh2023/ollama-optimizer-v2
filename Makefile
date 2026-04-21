.PHONY: install dev test lint format clean docker run-server run-mlflow run-langfuse

PYTHON := python3
VENV := .venv

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev,llmops]"

dev: install
	$(VENV)/bin/pre-commit install

test:
	$(VENV)/bin/pytest tests/ -v --cov=internal --cov=cmd --cov-report=term-missing

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy internal cmd

format:
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

docker:
	docker build -t ollama-optimizer:latest -f deploy/Dockerfile .

docker-compose-up:
	docker compose -f deploy/docker-compose.yml up -d

run-server:
	$(VENV)/bin/uvicorn cmd.server.main:app --reload --port 8000

run-mlflow:
	$(VENV)/bin/mlflow server --host 0.0.0.0 --port 5000

detect:
	$(VENV)/bin/ollama-opt detect

bench:
	$(VENV)/bin/ollama-opt bench --model llama3.2:3b

tune:
	$(VENV)/bin/ollama-opt tune --model llama3.2:3b
