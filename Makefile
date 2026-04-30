.PHONY: help install test format lint typecheck check clean dev-install run-example agent-tests gym gym-list gym-failures gym-security gym-agent \
        dogfood-agent dogfood-check dogfood-snapshot dogfood-run dogfood-mcp dogfood-healing dogfood-snapshot-core \
        dogfood-check-core dogfood-reporting dogfood-agent-docs dogfood-core

# Default target
help:
	@echo "EvalView Development Commands"
	@echo ""
	@echo "Setup (choose pip OR uv):"
	@echo "  make install       Install package (uv sync)"
	@echo "  make dev-install   Install with all extras (uv sync --all-extras)"
	@echo "  make pip-install   Install package (pip install -e .)"
	@echo "  make pip-dev       Install with dev extras (pip install -e '.[dev]')"
	@echo ""
	@echo "Development (uv - faster):"
	@echo "  make format        Format code with black"
	@echo "  make lint          Lint code with ruff"
	@echo "  make typecheck     Type check with mypy"
	@echo "  make check         Run all checks (format + lint + typecheck)"
	@echo "  make test          Run tests with pytest"
	@echo ""
	@echo "Development (pip - traditional):"
	@echo "  make pip-format    Format code with black (pip)"
	@echo "  make pip-lint      Lint code with ruff (pip)"
	@echo "  make pip-typecheck Type check with mypy (pip)"
	@echo "  make pip-check     Run all checks (pip)"
	@echo "  make pip-test      Run tests with pytest (pip)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Clean build artifacts and cache"
	@echo "  make run-example   Run example test case"
	@echo "  make agent-tests   Run EvalView agent tests (no CI required)"
	@echo ""
	@echo "Gym (Practice Evals):"
	@echo "  make gym           Run all gym scenarios"
	@echo "  make gym-list      List available scenarios"
	@echo "  make gym-failures  Run failure-mode scenarios only"
	@echo "  make gym-security  Run security scenarios only"
	@echo "  make gym-agent     Start the gym agent (localhost:2024)"
	@echo ""
	@echo "Internal Dogfooding:"
	@echo "  make dogfood-mcp            MCP server + contract slice"
	@echo "  make dogfood-healing        Healing policy slice"
	@echo "  make dogfood-snapshot-core  Snapshot/golden slice"
	@echo "  make dogfood-check-core     Check/diff/root-cause slice"
	@echo "  make dogfood-reporting      HTML/CLI reporting slice"
	@echo "  make dogfood-agent-docs     Agent-docs slice (plus manual review)"
	@echo "  make dogfood-core           Common core ship gate"
	@echo ""

# ============================================
# UV-based commands (faster, recommended)
# ============================================

install:
	uv sync

dev-install:
	uv sync --all-extras

format:
	@echo "Formatting code with black..."
	uv run black evalview/ tests/ --line-length 100

lint:
	@echo "Linting code with ruff..."
	uv run ruff check evalview/ tests/

typecheck:
	@echo "Type checking with mypy..."
	uv run mypy evalview/ --strict

check: format lint typecheck
	@echo "✅ All checks passed!"

test:
	@echo "Running tests with pytest..."
	@uv sync --all-extras --quiet
	uv run pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	uv run pytest tests/ --cov=evalview --cov-report=html --cov-report=term

# ============================================
# Pip-based commands (traditional)
# ============================================

pip-install:
	pip install -e .

pip-dev:
	pip install -e ".[dev]"

pip-format:
	@echo "Formatting code with black..."
	black evalview/ tests/ --line-length 100

pip-lint:
	@echo "Linting code with ruff..."
	ruff check evalview/ tests/

pip-typecheck:
	@echo "Type checking with mypy..."
	mypy evalview/ --strict

pip-check: pip-format pip-lint pip-typecheck
	@echo "✅ All checks passed!"

pip-test:
	@echo "Running tests with pytest..."
	pytest tests/ -v

pip-test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=evalview --cov-report=html --cov-report=term

# ============================================
# Shared utilities
# ============================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	rm -rf .venv/ venv/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

run-example:
	@echo "Running example test case..."
	@if [ ! -d ".evalview" ]; then \
		echo "Initializing EvalView..."; \
		uv run evalview init --dir .; \
	fi
	@if [ -f "tests/test-cases/example.yaml" ]; then \
		uv run evalview run --pattern "example.yaml" --verbose; \
	else \
		echo "❌ No example.yaml found. Run 'evalview init' first."; \
	fi

agent-tests:
	@echo "Running EvalView agent tests..."
	uv run evalview run --pattern "tests/test-cases/*.yaml" --verbose

dev: dev-install
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'uv run evalview init' to create a test project"
	@echo "  2. Make your changes"
	@echo "  3. Run 'make check' to verify code quality"
	@echo "  4. Run 'make test' to run tests"

quick-test:
	@echo "Running quick test (no coverage)..."
	uv run pytest tests/ -x --tb=short

# ============================================
# Gym - Agent Resilience Training
# ============================================

gym:
	@echo "━━━ EvalView Gym ━━━"
	@echo "Running all gym scenarios..."
	uv run evalview gym

gym-list:
	@echo "━━━ Gym Scenarios ━━━"
	uv run evalview gym --list-only

gym-failures:
	@echo "━━━ Running Failure Mode Scenarios ━━━"
	uv run evalview gym --suite failure-modes

gym-security:
	@echo "━━━ Running Security Scenarios ━━━"
	uv run evalview gym --suite security

gym-agent:
	@echo "━━━ Starting Gym Agent ━━━"
	@echo "Agent will run at http://localhost:2024"
	@echo ""
	cd gym/agents/support-bot && uv run langgraph dev

# ── Dogfood regression tests ──────────────────────────────────────────────────
# EvalView testing itself using the deterministic mock agent (port 8002).
# Tests the 3 correct-behavior scenarios to catch EvalView evaluation regressions.

## Start the deterministic mock agent (keep running in a separate terminal)
dogfood-agent:
	uv run python dogfood/mock_agent.py

## Check for regressions in EvalView's evaluation logic (requires dogfood-agent)
dogfood-check:
	uv run evalview check dogfood/mock-agent-tests/ --fail-on REGRESSION

## Save current evaluation results as the new golden baseline (requires dogfood-agent)
dogfood-snapshot:
	uv run evalview snapshot dogfood/mock-agent-tests/

## Run the full dogfood suite including failure-detection tests (requires dogfood-agent)
dogfood-run:
	uv run evalview run dogfood/mock-agent-tests/

## MCP slice — MCP server, contracts, and wrapper behavior
dogfood-mcp:
	uv run pytest -q tests/test_mcp_server.py tests/test_mcp_contracts.py

## Healing slice — healing engine and model/runtime recovery behavior
dogfood-healing:
	uv run pytest -q tests/test_healing.py tests/test_model_runtime_detector.py

## Snapshot slice — baseline creation, variants, and snapshot workflow
dogfood-snapshot-core:
	uv run pytest -q tests/test_snapshot_generated_workflow.py tests/test_e2e_snapshot_check.py tests/test_golden_store.py

## Check slice — check command, diffing, root-cause, and tag-filter behavior
dogfood-check-core:
	uv run pytest -q tests/test_check_cmd.py tests/test_check_pipeline.py tests/test_root_cause.py tests/test_run_cmd_tags.py

## Reporting slice — HTML reports, CI comments, and regression presentation
dogfood-reporting:
	uv run pytest -q tests/test_visualization_generators.py tests/test_ci_generate_comment.py

## Agent docs slice — lightweight automated guard plus manual review of agent-facing docs
dogfood-agent-docs:
	uv run pytest -q tests/test_new_features.py
	@echo ""
	@echo "Manual review:"
	@echo "  README.md"
	@echo "  AGENTS.md"
	@echo "  docs/agent-recipes/README.md"

## Common internal ship gate for broad core changes
dogfood-core:
	uv run pytest -q \
		tests/test_mcp_server.py \
		tests/test_mcp_contracts.py \
		tests/test_healing.py \
		tests/test_model_runtime_detector.py \
		tests/test_snapshot_generated_workflow.py \
		tests/test_e2e_snapshot_check.py \
		tests/test_golden_store.py \
		tests/test_check_cmd.py \
		tests/test_check_pipeline.py \
		tests/test_root_cause.py \
		tests/test_run_cmd_tags.py \
		tests/test_visualization_generators.py \
		tests/test_ci_generate_comment.py
