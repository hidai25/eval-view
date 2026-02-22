# EvalView Changelog

All notable changes to EvalView (the open-source AI agent testing framework) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4] - 2026-02-01

### Added

#### Agent-Based Skill Testing
- **Real agent execution**: Test skills through Claude Code CLI instead of just system prompts
- **Six agent adapters**: claude-code (primary), codex, langgraph, crewai, openai-assistants, custom
- **Two-phase evaluation**: Deterministic checks (Phase 1) + LLM rubric scoring (Phase 2)
- **Test categories**: explicit, implicit, contextual, negative (following OpenAI eval guidelines)

#### Phase 1 Deterministic Checks
- Tool checks: `tool_calls_contain`, `tool_calls_not_contain`, `tool_sequence`
- File checks: `files_created`, `files_modified`, `file_contains`, `file_not_contains`
- Command checks: `commands_ran`, `commands_not_ran`, `command_count_max`
- Output checks: `output_contains`, `output_not_contains`
- Token budget: `max_input_tokens`, `max_output_tokens`, `max_total_tokens`
- Build verification: `build_must_pass` - commands that must exit with code 0
- Smoke tests: `smoke_tests` with command/url/expected_output support
- Repository cleanliness: `git_clean` - verify no uncommitted changes
- Security checks: `no_sudo`, `forbidden_patterns`, `no_network_external`

#### CLI Enhancements
- `evalview skill test tests.yaml --agent claude-code` - run with real agents
- `--trace ./traces/` - capture JSONL traces for debugging
- `--no-rubric` - skip Phase 2 rubric evaluation
- `--cwd` and `--max-turns` options

#### Chat Mode Integration
- `/skill` slash command for interactive skill testing
- `/skill test`, `/skill validate`, `/skill list`, `/skill doctor` subcommands
- Comprehensive skill testing documentation in chat assistant

### Fixed
- Stream-JSON parsing for Claude Code CLI output
- Negative test evaluation logic for `should_trigger: false` tests
- Type safety in adapter and runner (mypy compliance)

## [0.2.3] - 2026-01-25

### Added

#### CLI Statistical Mode Flags
- **`--runs N` flag**: Run each test N times for statistical evaluation (2-100)
  - Enables pass@k reliability metrics without modifying YAML files
  - Overrides per-test variance configuration
- **`--pass-rate` flag**: Set required pass rate for `--runs` mode (default: 0.8)
- **`--difficulty` filter**: Filter tests by difficulty level (`trivial`, `easy`, `medium`, `hard`, `expert`)

#### Difficulty Levels for Test Cases
- **New `difficulty` field** on test cases: `trivial`, `easy`, `medium`, `hard`, `expert`
- Enables benchmark stratification and capability profiling
- Console reporter shows difficulty column and breakdown by difficulty level
- Filter tests by difficulty: `evalview run --difficulty hard`

#### Partial Credit for Sequence Evaluation
- **`progress_score` field** on SequenceEvaluation (0.0 to 1.0)
- Sequence scoring now uses partial credit instead of binary pass/fail
- Example: If expected sequence is `[a, b, c, d, e]` and agent completed `[a, b, c]`, progress_score = 0.6
- Contributes 60% of the sequence weight to overall score instead of 0%
- Console output shows progress percentage for incomplete sequences

### Fixed
- **`--runs` CLI flag**: Now properly implemented (was documented but missing in v0.1.5)

## [0.2.0] - 2026-01-10

### Added

#### Flexible Sequence Evaluation (Breaking Change)
- **Three sequence matching modes**: `subsequence` (new default), `exact`, and `unordered`
- `subsequence`: Expected tools must appear in order, but extra tools are allowed between them
- `exact`: Legacy strict matching - tools must match exactly in count and order
- `unordered`: Expected tools must be called, order doesn't matter
- Mode can be set via `SequenceEvaluator(default_mode=...)` or per-test in `adapter_config.sequence_mode`

#### Industry-Standard Reliability Metrics
- **pass@k**: Probability of at least one success in k trials ("will it work eventually?")
- **pass^k**: Probability of all k trials succeeding ("will it work reliably every time?")
- Both metrics now displayed in statistical evaluation summaries
- Color-coded interpretations: green/yellow/red based on reliability thresholds

#### Suite Type Classification
- **New `suite_type` field** on test cases: `capability` or `regression`
- `capability`: Tests measuring what the agent CAN do (expect lower pass rates, hill-climbing)
- `regression`: Tests verifying the agent STILL works (expect ~100% pass rate, safety net)
- Different status indicators in console output: `🚨 REGRESSION` vs `⚡ CLIMBING`
- Suite breakdown in summary panel with regression failure alerts

### Changed
- **Default sequence mode changed from `exact` to `subsequence`** - This prevents penalizing agents for finding valid alternative paths (per Anthropic's agent evaluation best practices)
- Console reporter now shows suite type column when tests have suite types defined
- Statistical comparison table includes pass@k and pass^k columns

### Fixed
- Sequence evaluator no longer fails tests when agents use additional intermediate tools

## [0.1.5] - 2025-12-19

### Added
- **Statistical Pass/Fail System**: Variance-aware testing with configurable confidence levels for more reliable evaluations
- **Statistical Mode in CLI**: New `--runs` flag to run tests multiple times and get statistical results
- **Templates**: Added test case templates for common scenarios

### Fixed
- **LangGraph Adapter**: Fixed adapter compatibility issues
- **Config-free runs**: Allow `evalview run` without requiring a config file
- **Node SDK License**: Fixed license mismatch - now correctly uses Apache 2.0

### Documentation
- Added FAQ section and comparison table to README
- Added "Run examples directly" section
- Added design partners section
- Improved README structure for better clarity

## [0.1.4] - 2025-12-15

### Added
- **Ollama Support**: Use Ollama as LLM-as-judge provider for free local evaluation
- **Ollama Adapter**: New adapter for testing LangGraph+Ollama agents
- Auto-detect Ollama when running locally

### Documentation
- Added Ollama example project with setup guide

## [0.1.3] - 2025-12-08

### Added
- **CLI guide for creating test cases**: After each run, shows inline YAML example with instructions
- **GPT-5 and Gemini 3.0 support**: Updated model aliases for latest models

### Fixed
- `Evaluator()` constructor in quickstart command (removed deprecated `openai_api_key` parameter)

### Documentation
- Updated TROUBLESHOOTING.md with Common Pitfalls section
- Added CLI flags documentation to README, GETTING_STARTED, and QUICKSTART_HUGGINGFACE
- Improved guidance on LLM-as-judge model selection

## [0.1.1] - 2025-12-05

### Added

#### HuggingFace Integration
- **HuggingFace as LLM-as-Judge**: Use open-source models (Llama, Mixtral) for evaluation instead of OpenAI—zero cost, full privacy
- **HuggingFace Spaces Adapter**: Test Gradio-based agents hosted on HuggingFace Spaces
- New environment variables: `EVAL_PROVIDER=huggingface`, `HF_TOKEN`, `EVAL_MODEL`
- Quick start guide for HuggingFace users (`docs/QUICKSTART_HUGGINGFACE.md`)

#### Developer Experience Improvements
- **CLI flags for LLM-as-Judge**: `--judge-model` and `--judge-provider` flags for easy model switching
- **Model shortcuts**: Use simple names like `gpt-5`, `sonnet`, `llama-70b` that auto-resolve to full model names
- **OpenAI Assistant Auto-Creation**: Automatically create an assistant with user confirmation when `OPENAI_ASSISTANT_ID` is not set
- Adapter aliases for convenience: `hf` and `gradio` map to HuggingFace adapter
- Example configurations for Anthropic and HuggingFace adapters

### Changed
- **Improved Hallucination Detector**: Reduced false positives by distinguishing between actual hallucinations (false facts, invented data) and helpful general advice (recommendations, best practices)
- Updated LLM provider to use HuggingFace's unified router endpoint (`router.huggingface.co`)

### Fixed
- OpenAI Assistants adapter no longer tracks `message_creation` as a tool step (was causing unexpected tool failures)
- Assistant ID persistence to `.env.local` for session continuity

### Documentation
- Added comprehensive HuggingFace quick start guide explaining Agent vs Judge concept
- Updated `ADAPTERS.md` with HuggingFace adapter documentation
- New example projects under `examples/huggingface/` and `examples/anthropic/`

## [0.1.0] - 2025-01-24

### Added

#### Core Framework
- CLI commands: `init`, `run`, and `report`
- Interactive project initialization with `evalview init --interactive`
- YAML-based test case format with comprehensive validation
- Execution trace capture for debugging and analysis
- JSON and rich console reporting

#### Adapters
- `HTTPAdapter` - Generic REST API adapter for any agent backend
- `LangGraphAdapter` - Native integration with LangGraph agents
- `CrewAIAdapter` - Support for CrewAI agents
- `OpenAIAssistantsAdapter` - OpenAI Assistants API integration
- `TapeScopeAdapter` - TapeScope framework support
- Custom adapter extensibility through base `AgentAdapter` class

#### Evaluators
- `ToolCallEvaluator` - Validates tool/function calls (30% weight)
- `SequenceEvaluator` - Checks tool call ordering (20% weight)
- `OutputEvaluator` - LLM-as-judge output quality assessment (50% weight)
- `CostEvaluator` - Cost threshold validation
- `LatencyEvaluator` - Latency threshold validation
- Per-test adapter configuration support
- Configurable evaluator weights

#### Testing Infrastructure
- Comprehensive test suite with 154+ tests across 22 test classes
- Test coverage for all core components
- Async testing support with pytest-asyncio
- Mock fixtures for OpenAI and HTTP adapters
- Test markers: `unit`, `integration`, `evaluator`, `adapter`

#### Development Tools
- Makefile with common development commands
- Black code formatting (100 char line length)
- Ruff linting configuration
- Mypy strict type checking
- Development mode installation support

#### Documentation
- Comprehensive README with quick start guide
- CONTRIBUTING.md with development workflow
- Architecture documentation
- Adapter development guide (ADAPTERS.md)
- Framework integration guide (FRAMEWORK_SUPPORT.md)
- Cost tracking documentation (COST_TRACKING.md)
- Debugging guide (DEBUGGING.md)
- Database setup guide (DATABASE_SETUP.md)

#### SDKs and Integrations
- Node.js SDK with Express/Next.js middleware
- Backend integration examples
- Stock analysis example test case

#### Features
- Case-insensitive test filtering with smart substring matching
- Detailed per-test reports with query and response
- Cost and latency tracking in execution traces
- Rich terminal output with color-coded results
- Verbose debug mode for troubleshooting
- Environment variable support for API keys

### Fixed
- Test case loader validation edge cases
- Async handling in HTTP adapters
- Type hints compatibility with Python 3.9+

### Security
- API key management via environment variables
- .gitignore configuration for sensitive files
- Input validation using Pydantic models

## [0.0.1] - Initial Development

### Added
- Initial project structure
- Basic CLI framework
- Core type definitions

---

## Release Types

- **Major version** (X.0.0): Breaking changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, backward compatible

## Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes and improvements

[Unreleased]: https://github.com/hidai25/eval-view/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/hidai25/eval-view/compare/v0.2.0...v0.2.3
[0.2.0]: https://github.com/hidai25/eval-view/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/hidai25/eval-view/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/hidai25/eval-view/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/hidai25/eval-view/compare/v0.1.1...v0.1.3
[0.1.1]: https://github.com/hidai25/eval-view/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/hidai25/eval-view/releases/tag/v0.1.0
