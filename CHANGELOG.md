# EvalView Changelog

All notable changes to EvalView (the open-source AI agent testing framework) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`evalview model-check`** — new command that detects silent drift in
  closed-weight models (Claude, GPT, ...) by running a small structural
  canary suite directly against the provider. Three-anchor comparison
  (reference / previous / trend), dry-run cost estimation, honest
  per-provider fingerprint strength labeling, custom suites via
  `--suite`, suite-hash enforcement for rotation safety. See
  `docs/MODEL_CHECK.md`.
- Bundled canary suite (`evalview/benchmarks/canary/suite.v1.public.yaml`):
  15 structural prompts across four scorer families (tool choice,
  JSON schema, refusal, exact match). Versioned, hash-pinned, and
  rotated via a held-out companion suite.
- `DriftKind` + `DriftConfidence` enums (`core/drift_kind.py`) — unified
  drift taxonomy orthogonal to `DiffStatus`. Reserved values for
  MCP contract drift so the planned P0 roadmap item does not refactor
  the taxonomy again.
- `core/model_snapshots.py` — timestamped snapshot store with auto-pin
  first-run reference, hash-enforced compatibility checks, and
  automatic pruning to the last 50 snapshots per model.
- `core/model_check_scoring.py` — pure-function structural scorers
  (`tool_choice`, `json_schema`, `refusal`, `exact_match`). No LLM
  judge dependency, no calibration requirement.
- `core/canary_suite.py` — loader and content hasher for canary YAMLs.
- `anthropic` adapter is now registered in `core/adapter_factory.py`
  so it can be driven from config or `evalview model-check`.
- New tests: 94 net new tests covering the snapshot store, scorers,
  canary loader, and command integration (all against mocked
  adapters — no real API calls in CI).

### Changed
- `TraceDiff` gains two optional fields: `drift_kind` and
  `drift_confidence`. Both default to `None`; existing callers are
  unaffected.
- `REFUSAL_PATTERNS` promoted to a public constant in
  `evalview/test_generation.py` so the model-check scorers can reuse
  the single source of truth without duplication. The legacy
  `_REFUSAL_PATTERNS` name is kept as an alias.

## [0.6.1] - 2026-03-29

### Added
- **Full MCP feature parity** — all CLI flags now exposed via MCP: heal, strict, ai_root_cause, statistical, auto_variant, budget, dry_run, tag, fail_on, timeout, report, judge on run_check; variant, preview, reset on run_snapshot; new compare_agents and replay tools
- **MCP server regression tests** — 33 test cases covering protocol, schema contracts, flag wiring, routing, timeouts, and error handling

### Fixed
- MCP run_check subprocess path now forces `--json` for stable response contract
- MCP server sets `CI=1` to prevent `--report` from opening browser
- MCP replay timeout increased from 30s to 120s
- MCP subprocess calls use `stdin=DEVNULL` to prevent hangs

## [0.6.0] - 2026-03-27

### Added
- **Auto-heal engine** — `evalview check --heal` automatically retries flaky tests, distinguishes non-determinism from real regressions, and self-heals output drift
- **Model change detection** — detects when the underlying model has changed and adjusts evaluation accordingly

## [0.5.5] - 2026-03-25

### Added
- **`evalview watch`** — file watcher that re-runs regression checks on every save with live scorecard dashboard, debounce, directory exclusions, and `--quick` mode ($0, sub-second)
- **`evalview badge`** — generate shields.io-compatible status badge JSON; auto-updates on every `evalview check`
- **`evalview monitor --dashboard`** — live-updating Rich terminal dashboard with per-test status, history dots, cycle count, uptime, cost tracking, and Slack alert count
- **Native Pydantic AI adapter** (`pydantic-ai`) — runs `agent.run()` in-process, extracts tool calls from typed `ModelResponse`/`ToolCallPart` messages, captures usage from `result.usage()`
- **Native CrewAI adapter** (`crewai-native`) — runs `crew.kickoff()` in-process, captures tool calls via event bus (`ToolUsageFinishedEvent`) with exact args, output, agent role, and timing
- **Assertion wizard** — `evalview capture` now analyzes captured traffic and suggests smart assertions (tool sequence, required tools, latency, quality score) automatically
- **Auto-variant discovery** — `evalview check --statistical N --auto-variant` runs tests N times, clusters execution paths, and saves distinct valid paths as golden variants
- **Budget circuit breaker** — `evalview check --budget 0.50` enforces spend limits mid-execution with per-test cost breakdown; remaining tests skipped when limit hit
- **Smart eval profiles** — `evalview init` auto-detects agent type (chat, tool-use, multi-step, RAG, coding) and pre-configures evaluators, thresholds, and assertions
- **Python API** — `gate()`, `gate_async()`, `gate_or_revert()` for programmatic regression checks; used by watch mode, MCP server, and OpenClaw integration
- **OpenClaw integration** — `evalview openclaw install` and `evalview openclaw check` for autonomous agent loops with `check_and_decide()` and `gate_or_revert()` helpers
- **curl installer** — `curl -fsSL .../install.sh | bash` for zero-friction onboarding
- **GitHub Action improvements** — auto PR comments, artifact uploads, version pinning, job summary — all in one step

### Changed
- README rewritten for conversion: pain-first positioning, "Playwright for AI agents" analogy, tighter quickstart, promoted CI/watch/multi-turn sections
- HTML report: Health Gauge and Score Per Test now display side by side when no trend data available
- MCP server rewired to use `gate()` API instead of subprocess

### Fixed
- Check command test mocks updated for `budget_tracker` parameter
- Badge command mypy type errors resolved

### Documentation
- CrewAI integration guide with multi-agent and safety test examples
- Pydantic AI integration guide with FastAPI wrapper and comparison to pydantic_evals
- OpenClaw walkthrough with prompt optimizer example
- 26 community issues created (8 good-first-issue, 15 help-wanted, 3 feature requests)
- GitHub Actions example simplified to 6-line workflow

## [0.5.1] - 2026-03-13

### Added
- **`evalview generate`** — draft test suite generation from agent probing or log imports, with approval gating and CI review flow
- **Approval workflow** — generated tests require explicit approval before becoming baselines
- **CI review comments** — `evalview ci comment` posts generation reports on PRs

### Fixed
- Python 3.9 compatibility: replaced `datetime.UTC` with `timezone.utc`
- Mypy type errors in generate command and test generation module
- Codebase refactor and cleanup across 71 files

## [0.5.0] - 2026-03-12

### Added
- **`evalview monitor`** — continuous regression detection for production. Runs `evalview check` in a loop with configurable interval, graceful Ctrl+C shutdown, and cumulative cost tracking
- **Slack alerts** — webhook notifications on new regressions with smart dedup (no re-alerts on persistent failures) and recovery notifications when issues are resolved
- **`--history` flag** — append each monitor cycle's results to a JSONL file for trend analysis and dashboards (community contribution by @clawtom)
- **`--csv` flag for check** — export check results to CSV (community contribution by @muhammadrashid4587)
- **`--timeout` flag for check** — configurable per-test timeout with validation (community contribution by @zamadye)
- **Better error messages** — human-friendly connection failure messages with actionable guidance (community contribution by @passionworkeer)
- **Monitor config** — `MonitorConfig` model with validation, configurable via CLI flags, `config.yaml`, or `EVALVIEW_SLACK_WEBHOOK` env var
- **Playful monitor messages** — rotating start, cycle, and success messages to prevent repetition fatigue
- **Deployment docs** — nohup, systemd, and Docker examples for running monitor as a background service

### Fixed
- Severity comparison bug in JSONL history — was using fail_on filter instead of actual pass/fail counts
- DiffStatus comparisons now use type-safe enum comparison instead of fragile string matching
- Mypy type error in CSV export
- Redundant config loading in monitor loop eliminated

### Changed
- Extracted `_parse_fail_statuses` shared utility for consistent fail_on parsing across check and monitor commands
- Monitor loop receives config from CLI handler instead of loading it twice

### Community
- 4 community PRs merged in this release — thank you @clawtom, @muhammadrashid4587, @zamadye, and @passionworkeer!

## [0.4.1] - 2026-03-09

### Added
- **Mistral adapter** — direct Mistral API support via `pip install evalview[mistral]`
- **PII evaluator** — opt-in detection for emails, phones, SSNs, credit cards, addresses (with Luhn validation)
- **Multi-turn HTML reports** — Mermaid sequence diagrams showing conversation turns with tool calls

### Fixed
- GitHub Action security: replaced `eval $CMD` with bash arrays, moved inputs to env vars
- Mermaid diagram rendering: fixed autoescape breaking arrows, sanitized user content
- Multi-turn step tracing: annotated steps with `turn_index` and `turn_query`

### Changed
- README hero section rewritten for clarity — logo, sequence diagram hero, data flow explanation
- Model version examples updated to Claude 4.5/4.6

## [0.4.0] - 2026-03-05

### Added

#### Multi-Turn Conversation Testing
- **`turns:` YAML field** — replace `input` with a `turns` list to test stateful, multi-step conversations
- **`ConversationTurn` model** — each turn has `query`, optional `expected`, and optional `context`
- **Automatic history injection** — accumulated `conversation_history` is passed in each turn's context so agents can track context across turns
- **Per-turn `expected` assertions** — tool checks and output assertions scoped to each turn
- **Merged trace** — all turns' tool calls, costs, and latency are summed into a single `ExecutionTrace` for evaluation
- **Backward compatibility** — `input` is auto-populated from the first turn so all downstream code works unchanged

#### A/B Endpoint Comparison
- **`evalview compare` command** — run the same test suite against two endpoints simultaneously
- **Per-test verdict table** — `improved` / `degraded` / `same` with score delta for each test
- **Overall verdict panel** — production-ready guidance (promote / caution / investigate)
- **`--label-v1` / `--label-v2`** — human-readable labels appear in output and report filenames
- **`--no-judge`** — skip LLM judge for fast, cost-free deterministic comparison
- **`--no-open`** — suppress auto-opening HTML report (CI-friendly)

#### Cloud Baseline Sync
- **`evalview login`** — OAuth-based authentication, opens browser for sign-in
- **`evalview logout`** — clear local credentials
- **`evalview whoami`** — show currently authenticated user
- **Silent push after snapshot** — golden baselines automatically sync to cloud after every `evalview snapshot`
- **Silent pull before check** — remote baselines pulled before every `evalview check` so teammates always compare against the team baseline

#### evalview capture
- **HTTP proxy recorder** — `evalview capture --agent <url>` starts a proxy on `:8091` that records real agent traffic as test YAML files
- **Zero-config** — point your app at the proxy, use it normally, Ctrl-C to stop; test files are written automatically
- **Custom port** — `--port 8092` for non-default proxy ports
- **Best-practice onboarding** — capture is now the first recommended step for new projects

#### Silent Regression Detection
- **Model fingerprinting** — captures model version at snapshot time; alerts when provider silently swaps the model behind the same API name
- **Gradual drift detection** — OLS regression over a 10-check window catches slow similarity decline that single-threshold checks miss
- **Semantic diff** — `--semantic-diff` uses OpenAI embeddings to score outputs by meaning, not character-level similarity

#### Git Hook Integration
- **`evalview install-hooks`** — injects `evalview check` into your repo's pre-push (or pre-commit) hook; automatic regression blocking with zero CI config
- **`evalview uninstall-hooks`** — cleanly removes installed hooks

#### Auto-Open HTML Report
- Every `evalview run` automatically opens the interactive HTML report in your browser
- `--no-open` flag suppresses this for CI pipelines

#### evalview init Improvements
- **Auto-detect agent endpoint** — probes common ports to find your running agent automatically
- **Auto-generate test cases** — probes the agent with representative queries to create a starter test suite
- **Context-aware completion panel** — shows detected agent URL, model, and generated test count

#### Test Quality Gating
- **Quality score per test** — generated tests are scored before execution; low-quality tests are skipped with a warning rather than polluting agent scores
- **Quality hints** — when tests look like the problem (not the agent), EvalView says so clearly

### Fixed
- **Mermaid parameter display** — `offset=0, limit=10` now renders correctly (was `offset0 limit10`)
- **Duplicate banner in demo** — `evalview demo` no longer shows the banner twice
- **Cloud sync noise** — "Cloud sync skipped (offline?)" suppressed during demo mode
- **mypy errors** — `TestCase.input` restored as required field using a `mode="before"` validator; fixes 13 call-site errors across 4 files
- **False REGRESSION on identical output** — judge score variance on identical output no longer triggers a regression
- **Zero/negative thresholds** — `CostEvaluator` and `LatencyEvaluator` now handle 0 and negative threshold values correctly
- **Agent endpoint detection** — checks for `output` field in POST response, not just HTTP 200

### Community
- Pydantic field validation for `TestCase` model (#54 by @illbeurs)
- Edge test cases for `CostEvaluator` and `LatencyEvaluator` (#55 by @illbeurs)
- `health_check()` method on `OllamaAdapter` (#57 by @gauravxthakur)
- `ConsoleReporter` docstrings (#56 by @gauravxthakur)

---

## [0.3.2] - 2026-02-27

### Added
- **DeepSeek provider** — DeepSeek is now a first-class LLM judge provider alongside OpenAI and Anthropic
- **Glama MCP registry** — EvalView MCP server listed on Glama (`glama.json` + `Dockerfile`)

### Fixed
- Custom runner + MCP timeout handling for OAuth-authenticated users
- Nested Claude auth failure in `claude-code` adapter
- Graceful LLM fallback when primary provider is unavailable

---

## [0.3.1] - 2026-02-25

### Added

#### Visual Reports
- **Glassmorphism HTML reports** — interactive reports with Plotly charts, Mermaid sequence diagrams, cost-per-query table, and full query/response in trace view
- **`evalview inspect`** — open an HTML report from any result file
- **`generate_visual_report` MCP tool** — generate reports inline from Claude Code

#### Safety & Forensics
- **`forbidden_tools` safety contracts** — declare tools that must never be called; any violation is an immediate hard-fail (score 0, no partial credit)
- **HTML trace replay** — step-by-step replay of every LLM call and tool invocation with exact prompt, completion, tokens, and parameters

#### LLM Judge Caching
- **Judge response cache** — cache LLM judge responses during repeated test runs; ~80% fewer API calls in statistical mode
- Stored in `.evalview/.judge_cache.db` (SQLite)

#### Skills Testing
- **15-template pattern library** — `evalview add <pattern>` copies ready-made YAML patterns to your project
- **Personalized init wizard** — `evalview init --wizard` generates a config and first test tailored to your agent in 3 questions
- **Provider-agnostic skill tests** — run skill tests against Anthropic, OpenAI, DeepSeek, or any OpenAI-compatible API
- **OpenClaw adapter** — support for AgentSkills/SKILL.md testing
- **Security evaluator** — deterministic security checks in skill evaluation

### Fixed
- Resolved all mypy type errors in `skills/runner.py` and `visualization/generators.py`
- `--no-judge` flag and fail-fast API key validation on `evalview run`
- EvalView banner now shown on demo, run, and snapshot commands

---

## [0.3.0] - 2026-02-20

### Added

#### Claude Code MCP Integration
- **MCP server** — EvalView runs as an MCP server inside Claude Code
- **Skill testing tools** — `validate_skill`, `generate_skill_tests`, `run_skill_test` available as MCP tools
- **`create_test` MCP tool** — generate test cases from natural language without writing YAML

#### evalview demo
- **Live regression demo** — `evalview demo` runs a self-contained ~30-second regression demonstration with no setup required
- Customer support scenario with intentional regression injection and recovery

#### Agent Discovery
- **Early unreachable-agent detection** — connectivity check before any test output; clear onboarding guide shown when no agent is found
- **Agent endpoint shown in run output** — always visible which endpoint is under test

#### Telemetry & Observability
- Readable event names, person identification, session duration tracking
- Dev-mode filter to exclude local development runs

### Fixed
- Tool names shown in diff output
- Pluralization in check output
- Demo mode no longer shows "coming soon" noise

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
