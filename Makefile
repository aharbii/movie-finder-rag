# =============================================================================
# Movie Finder RAG — Docker-only developer contract
#
# Usage:
#   make help
#   make <target>
#
# All developer commands execute through Docker Compose so linting, testing,
# formatting, and pre-commit do not depend on a host-managed Python environment.
# Remote vector stores stay external; local ChromaDB uses the mounted workspace path.
#
# Typical first-time flow:
#   make init        # build images + create .env + install git hook
#   make editor-up   # start container for VS Code attach
#   make check       # lint + typecheck + tests with coverage
#
# When the editor container is already running, quality commands use
# 'docker compose exec' instead of a new container — faster for interactive dev.
# =============================================================================

.PHONY: help init build setup clean clean-docker \
	editor-up editor-down ci-down shell logs up down  run run-dev \
	lint format fix typecheck test test-coverage pre-commit detect-secrets check \
	backup cost-report qdrant-live-eval retrieve validate migrate-legacy-qdrant-collection tui


.DEFAULT_GOAL := help

COMPOSE ?= docker compose
SERVICE ?= rag
INGEST_SERVICE ?= ingestion
GIT_DIR_HOST := $(shell git rev-parse --git-dir)
GIT_HOOKS_DIR := $(GIT_DIR_HOST)/hooks

# Export so docker compose picks it up automatically (avoids per-command prefix).
export RAG_GIT_DIR := $(GIT_DIR_HOST)

SOURCE_PATHS := .
COVERAGE_XML ?= reports/coverage.xml
COVERAGE_HTML ?= reports/htmlcov
JUNIT_XML ?= reports/junit.xml
BACKUP_ARGS ?=
QDRANT_EVAL_ARGS ?=
MIGRATE_ARGS ?=

# ---------------------------------------------------------------------------
# exec when running, run --rm otherwise — avoids container startup overhead
# for interactive development while remaining correct for CI.
# ---------------------------------------------------------------------------
define exec_or_run
	@if $(COMPOSE) ps --services --status running 2>/dev/null | grep -qx "$(SERVICE)"; then \
		$(COMPOSE) exec $(SERVICE) $(1); \
	else \
		$(COMPOSE) run --rm --no-deps $(SERVICE) $(1); \
	fi
endef

help:
	@echo ""
	@echo "Movie Finder RAG — available targets"
	@echo "===================================="
	@echo ""
	@echo "  Setup"
	@echo "    init           Build images, create .env from template, install git hook"
	@echo ""
	@echo "  Editor"
	@echo "    editor-up      Start the attached-container workspace in the background"
	@echo "    editor-down    Stop the local workspace container"
	@echo "    shell          Open a bash shell in the workspace container"
	@echo ""
	@echo "  Lifecycle"
	@echo "    up             Alias for editor-up"
	@echo "    down           Alias for editor-down"
	@echo "    logs           Follow workspace container logs"
	@echo "    ci-down        Full cleanup for CI: stop containers and remove volumes"
	@echo ""
	@echo "  Quality"
	@echo "    lint           Run ruff check (report only)"
	@echo "    format         Run ruff format (apply)"
	@echo "    fix            Run ruff check --fix + ruff format (apply all auto-fixes)"
	@echo "    typecheck      Run mypy"
	@echo "    test           Run pytest"
	@echo "    test-coverage  Run pytest with coverage + JUnit output"
	@echo "    detect-secrets Run detect-secrets scan"
	@echo "    pre-commit     Run all pre-commit hooks"
	@echo "    check          lint + typecheck + test-coverage"
	@echo ""
	@echo "  Maintenance"
	@echo "    clean          Remove __pycache__, .pytest_cache, .mypy_cache, reports (via Docker)"
	@echo "    clean-docker   Stop containers and remove volumes"
	@echo ""
	@echo "  Compatibility aliases"
	@echo "    build          Alias for init"
	@echo "    run / run-dev  Alias for editor-up"
	@echo "    setup          Alias for init"
	@echo ""
	@echo "  Pipeline"
	@echo "    ingest         Run the one-shot ingestion pipeline against the configured vector store"
	@echo ""
	@echo "  Apps"
	@echo "    backup         Runs the Docker-backed backup utility and writes artifacts under outputs/"
	@echo "    cost-report    Refreshes outputs/reports/cost-report.json from ingestion outputs"
	@echo "    qdrant-live-eval  Evaluates existing Qdrant collections and writes HTML/JSON reports"
	@echo "    retrieve       Runs the interactive CLI to validate retrieval logic"
	@echo "    tui            Launches the full Textual TUI for retrieval evaluation"
	@echo "    validate       Runs the post-ingest validation script"
	@echo "    migrate-legacy-qdrant-collection  Backs up and migrates a legacy Qdrant collection into the ADR-style name"
	@echo ""

init:
	@if [ ! -f .env ]; then cp .env.example .env && echo ">>> .env created from .env.example"; fi
	$(COMPOSE) build $(SERVICE) $(INGEST_SERVICE)
	@printf '#!/bin/sh\nexec make pre-commit\n' > $(GIT_HOOKS_DIR)/pre-commit
	@chmod +x $(GIT_HOOKS_DIR)/pre-commit
	@echo ">>> git pre-commit hook installed (calls 'make pre-commit' on every commit)"

setup: init
build: init

editor-up:
	$(COMPOSE) up -d $(SERVICE)

up: editor-up
run: editor-up
run-dev: editor-up

editor-down:
	$(COMPOSE) down --remove-orphans

down: editor-down

ci-down:
	$(COMPOSE) down -v --remove-orphans

logs:
	$(COMPOSE) logs -f $(SERVICE)

shell:
	@if $(COMPOSE) ps --services --status running 2>/dev/null | grep -qx "$(SERVICE)"; then \
		$(COMPOSE) exec $(SERVICE) bash; \
	else \
		$(COMPOSE) run --rm $(SERVICE) bash; \
	fi

lint:
	$(call exec_or_run,ruff check $(SOURCE_PATHS))

format:
	$(call exec_or_run,ruff format $(SOURCE_PATHS))

fix:
	$(call exec_or_run,ruff check --fix $(SOURCE_PATHS))
	$(call exec_or_run,ruff format $(SOURCE_PATHS))

typecheck:
	$(call exec_or_run,mypy $(SOURCE_PATHS))

test:
	$(call exec_or_run,pytest tests/ -v --tb=short)

test-coverage:
	$(call exec_or_run,sh -c 'mkdir -p reports && pytest tests/ -v --tb=short \
		--junitxml=$(JUNIT_XML) \
		--cov=rag \
		--cov-branch \
		--cov-report=term-missing \
		--cov-report=xml:$(COVERAGE_XML) \
		--cov-report=html:$(COVERAGE_HTML)')

detect-secrets:
	$(call exec_or_run,detect-secrets scan --baseline .secrets.baseline) # pragma: allowlist secret

pre-commit:
	$(call exec_or_run,pre-commit run --all-files)

check: lint typecheck test-coverage

clean:
	@echo ">>> Removing Python cache files (via Docker)..."
	$(call exec_or_run,find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true)
	$(call exec_or_run,find . -type d -name ".pytest_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true)
	$(call exec_or_run,find . -type d -name ".mypy_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true)
	$(call exec_or_run,find . -type d -name ".ruff_cache" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true)
	$(call exec_or_run,find . -name "*.egg-info" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true)
	$(call exec_or_run,rm -rf reports/)
	$(call exec_or_run,rm -rf outputs/)
	$(call exec_or_run,rm -f ingestion-outputs.env)
	$(call exec_or_run,rm -rf dataset/*)
	@echo "Clean complete."

clean-docker: ci-down

ingest:
	$(COMPOSE) run --rm $(INGEST_SERVICE)

backup:
	$(call exec_or_run,python scripts/backup_vectorstore.py $(BACKUP_ARGS))

cost-report:
	$(call exec_or_run,python scripts/generate_cost_report.py)

qdrant-live-eval:
	$(call exec_or_run,python scripts/evaluate_qdrant_collections.py $(QDRANT_EVAL_ARGS))

retrieve:
	$(call exec_or_run,python scripts/retrieve.py)

tui:
	$(call exec_or_run,python scripts/launch_tui.py)

validate:
	$(call exec_or_run,python scripts/validate_ingestion.py)

migrate-legacy-qdrant-collection:
	$(call exec_or_run,python scripts/migrate_legacy_qdrant_collection.py $(MIGRATE_ARGS))
