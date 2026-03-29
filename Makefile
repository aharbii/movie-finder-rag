# =============================================================================
# movie-finder-rag — Docker-only developer contract
#
# Usage:
#   make help
#   make <target>
#
# All supported developer commands execute through Docker Compose so local
# linting, testing, formatting, typecheck, and pre-commit do not depend on a
# host-managed Python environment. Qdrant is always external.
# =============================================================================

.PHONY: help setup init up down editor-up editor-down logs shell lint format \
	typecheck test test-coverage detect-secrets pre-commit ingest check \
	build run run-dev ci-down clean

.DEFAULT_GOAL := help

COMPOSE ?= docker compose
SERVICE ?= rag
INGEST_SERVICE ?= ingestion
RAG_GIT_DIR_HOST := $(shell git rev-parse --git-dir)
SOURCE_PATHS := src tests scripts
COVERAGE_XML ?= coverage.xml
COVERAGE_HTML ?= htmlcov
JUNIT_XML ?= test-results.xml

help:
	@echo ""
	@echo "movie-finder-rag — available targets"
	@echo "===================================="
	@echo ""
	@echo "  Setup"
	@echo "    setup          First-time dev setup (build + .env)"
	@echo ""
	@echo "  Editor"
	@echo "    editor-up      Start the attached-container workspace in the background"
	@echo "    editor-down    Stop the local workspace container"
	@echo "    shell          Open a shell in the workspace container"
	@echo ""
	@echo "  Lifecycle"
	@echo "    init           Build the dev and ingestion images"
	@echo "    up             Alias for editor-up"
	@echo "    down           Alias for editor-down"
	@echo "    logs           Follow workspace container logs"
	@echo "    ci-down        Full cleanup for CI: stop containers and remove volumes"
	@echo ""
	@echo "  Quality"
	@echo "    lint           Run ruff check inside Docker"
	@echo "    format         Run ruff format inside Docker"
	@echo "    typecheck      Run mypy inside Docker"
	@echo "    test           Run pytest inside Docker"
	@echo "    test-coverage  Run pytest with coverage + JUnit output"
	@echo "    detect-secrets Run detect-secrets scan inside Docker"
	@echo "    pre-commit     Run pre-commit hooks inside Docker"
	@echo "    check          Convenience alias: lint + typecheck + test"
	@echo ""
	@echo "  Pipeline"
	@echo "    ingest         Run the one-shot ingestion pipeline against external Qdrant"
	@echo ""
	@echo "  Maintenance"
	@echo "    clean          Remove __pycache__, .pytest_cache, .mypy_cache, reports"
	@echo ""
	@echo "  Compatibility aliases"
	@echo "    build          Alias for init"
	@echo "    run            Alias for editor-up"
	@echo "    run-dev        Alias for editor-up"
	@echo ""

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ">>> .env created from .env.example. Fill in your keys before running."; \
	fi
	$(MAKE) init
	@echo ""
	@echo ">>> Setup complete. Run 'make editor-up' to start developing."

init:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) build $(SERVICE) $(INGEST_SERVICE)

up: editor-up

down: editor-down

editor-up:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) up -d $(SERVICE)

editor-down:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) down --remove-orphans

ci-down:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) down -v --remove-orphans

logs:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) logs -f $(SERVICE)

shell:
	@if RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) ps --services --status running | grep -qx "$(SERVICE)"; then \
		RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) exec $(SERVICE) sh; \
	else \
		RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm $(SERVICE) sh; \
	fi

lint:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm --no-deps $(SERVICE) ruff check $(SOURCE_PATHS)

format:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm --no-deps $(SERVICE) ruff format $(SOURCE_PATHS)

typecheck:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm --no-deps $(SERVICE) mypy $(SOURCE_PATHS)

test:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm --no-deps $(SERVICE) pytest tests/ -v --tb=short

test-coverage:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm --no-deps $(SERVICE) pytest tests/ -v --tb=short \
		--junitxml=$(JUNIT_XML) \
		--cov=rag \
		--cov-report=term-missing \
		--cov-report=xml:$(COVERAGE_XML) \
		--cov-report=html:$(COVERAGE_HTML)

detect-secrets:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm --no-deps $(SERVICE) detect-secrets scan --baseline .secrets.baseline

pre-commit:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm --no-deps $(SERVICE) pre-commit run --all-files

ingest:
	RAG_GIT_DIR="$(RAG_GIT_DIR_HOST)" $(COMPOSE) run --rm $(INGEST_SERVICE)

check: lint typecheck test

clean:
	@echo ">>> Removing Python cache files..."
	find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "coverage.xml" -not -path "./.git/*" -delete 2>/dev/null || true
	find . -name "test-results.xml" -not -path "./.git/*" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

build: init

run: editor-up

run-dev: editor-up
