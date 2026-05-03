# Aegis — top-level dev tasks.
#
# Default `make setup` installs the LOCAL/EVAL stack only (no Unsloth,
# no bitsandbytes — those are Linux/CUDA only). Training runs on Colab,
# which uses `make setup-train`.

.PHONY: help setup setup-eval setup-train download lint format test test-cov clean

help:
	@echo "Aegis make targets:"
	@echo "  setup         — alias for setup-eval (local / Windows-friendly)"
	@echo "  setup-eval    — install eval/baseline/CLI deps + spaCy model"
	@echo "  setup-train   — Colab-only: install training stack (Unsloth, bnb)"
	@echo "  download      — fetch ai4privacy/pii-masking-300k into data/raw/"
	@echo "  lint          — ruff check + format check"
	@echo "  format        — ruff format + autofix"
	@echo "  test          — pytest"
	@echo "  test-cov      — pytest with coverage report"
	@echo "  clean         — remove caches and build artifacts"

setup: setup-eval

setup-eval:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .
	python -m spacy download en_core_web_lg

setup-train:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -r requirements-train.txt
	python -m pip install -e .

download:
	python -m data.download

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

test-cov:
	pytest --cov=data --cov=train --cov=inference --cov=baselines --cov=eval --cov=cli --cov-report=term-missing

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.ruff_cache','.pytest_cache','build','dist','.coverage','htmlcov']]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.egg-info') if p.is_file()]"
