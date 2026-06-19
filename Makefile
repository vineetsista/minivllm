# Developer shortcuts. On Windows, run the underlying commands directly or use
# `make` via Git Bash / WSL.

.PHONY: install lint format typecheck test test-all bench serve clean

install:  ## install the package + dev tooling (CPU torch)
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,plot]"
	pre-commit install

lint:  ## ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## auto-format and auto-fix
	ruff check . --fix
	ruff format .

typecheck:  ## mypy
	mypy minivllm

test:  ## fast tests only (no model download)
	pytest

test-all:  ## include slow HF-reference gates (downloads Qwen3-0.6B)
	pytest --runslow

bench:  ## regenerate the benchmark numbers
	python -m scripts.bench_all

serve:  ## run the OpenAI-compatible server on :8000
	python -m uvicorn minivllm.server:app --port 8000

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache *.egg-info build dist
