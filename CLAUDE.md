# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AgentEval** - A Python-based CLI tool for testing and evaluating multi-step AI agents. Think "Playwright for AI agents."

The framework allows developers to write YAML-based test cases, execute them against AI agents via REST APIs, and get comprehensive evaluation reports covering tool usage, output quality, cost, and latency.

## Development Setup

```bash
# Create virtual environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or venv\Scripts\activate on Windows

# Install in development mode
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Common Commands

### Running the CLI
```bash
# Initialize a new project
agent-eval init

# Run test cases
agent-eval run

# Generate report
agent-eval report .agenteval/results/TIMESTAMP.json
```

### Development
```bash
# Format code
black agent_eval/

# Type checking
mypy agent_eval/

# Linting
ruff agent_eval/

# Run tests (when added)
pytest
```

### Building
```bash
# Install/reinstall after changes
pip install -e .
```

## Architecture

### Core Components

1. **CLI (`agent_eval/cli.py`)** - Click-based command-line interface
   - `init` - Initialize project structure
   - `run` - Execute test cases
   - `report` - Generate reports from results

2. **Core Types (`agent_eval/core/types.py`)** - Pydantic models
   - `TestCase` - Test case definition from YAML
   - `ExecutionTrace` - Agent execution capture
   - `EvaluationResult` - Complete evaluation output

3. **Adapters (`agent_eval/adapters/`)** - Agent communication
   - `AgentAdapter` - Abstract base class
   - `HTTPAdapter` - Generic REST API adapter

4. **Evaluators (`agent_eval/evaluators/`)** - Evaluation logic
   - `ToolCallEvaluator` - Tool accuracy (30% weight)
   - `SequenceEvaluator` - Tool sequence correctness (20% weight)
   - `OutputEvaluator` - LLM-as-judge output quality (50% weight)
   - `CostEvaluator` - Cost threshold checking
   - `LatencyEvaluator` - Latency threshold checking
   - `Evaluator` - Main orchestrator

5. **Reporters (`agent_eval/reporters/`)** - Result formatting
   - `JSONReporter` - JSON file output
   - `ConsoleReporter` - Rich terminal output

### Data Flow

1. User writes YAML test case → `TestCaseLoader`
2. `HTTPAdapter` executes agent and captures trace
3. `Evaluator` runs all sub-evaluators
4. Results saved as JSON and displayed in console

### Key Design Patterns

- **Adapter Pattern**: `AgentAdapter` abstracts different agent implementations
- **Strategy Pattern**: Multiple evaluators with pluggable logic
- **Model-View Pattern**: Core models separate from reporters

## Important Notes

- All type annotations use Python 3.9 compatible syntax (`List[str]` not `list[str]`)
- Async/await used for HTTP calls and LLM-as-judge
- Pydantic models handle validation and serialization
- OpenAI API key required for LLM-as-judge evaluation
- Test cases use YAML for human readability

## Environment Variables

- `OPENAI_API_KEY` - Required for output quality evaluation (LLM-as-judge)

## Repository Information

- **Git Repository**: https://github.com/hidai25/EvalView.git
- **Main Branch**: main
- **Package Name**: agent-eval
- **CLI Command**: agent-eval
