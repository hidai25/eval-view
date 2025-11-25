.PHONY: help install test format lint typecheck check clean dev-install run-example agent-tests

# Default target
help:
	@echo "EvalView Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package in development mode"
	@echo "  make dev-install   Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format        Format code with black"
	@echo "  make lint          Lint code with ruff"
	@echo "  make typecheck     Type check with mypy"
	@echo "  make check         Run all checks (format + lint + typecheck)"
	@echo "  make test          Run tests with pytest"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Clean build artifacts and cache"
	@echo "  make run-example   Run example test case"
	@echo "  make agent-tests   Run EvalView agent tests (no CI required)"
	@echo ""

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# Code quality
format:
	@echo "Formatting code with black..."
	black evalview/ tests/ --line-length 100

lint:
	@echo "Linting code with ruff..."
	ruff check evalview/ tests/

typecheck:
	@echo "Type checking with mypy..."
	mypy evalview/ --strict

# Run all checks
check: format lint typecheck
	@echo "✅ All checks passed!"

# Testing
test:
	@echo "Running tests with pytest..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=evalview --cov-report=html --cov-report=term

# Clean up
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

# Example workflow
run-example:
	@echo "Running example test case..."
	@if [ ! -d ".evalview" ]; then \
		echo "Initializing EvalView..."; \
		evalview init --dir .; \
	fi
	@if [ -f "tests/test-cases/example.yaml" ]; then \
		evalview run --pattern "example.yaml" --verbose; \
	else \
		echo "❌ No example.yaml found. Run 'evalview init' first."; \
	fi

# Agent tests - run EvalView against your agent (no CI required)
agent-tests:
	@echo "Running EvalView agent tests..."
	evalview run --pattern "tests/test-cases/*.yaml" --verbose

# Development helpers
dev: dev-install
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'evalview init' to create a test project"
	@echo "  2. Make your changes"
	@echo "  3. Run 'make check' to verify code quality"
	@echo "  4. Run 'make test' to run tests"

# Quick test during development
quick-test:
	@echo "Running quick test (no coverage)..."
	pytest tests/ -x --tb=short
