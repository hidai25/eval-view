# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-12-05

### Added

#### HuggingFace Integration
- **HuggingFace as LLM-as-Judge**: Use open-source models (Llama, Mixtral) for evaluation instead of OpenAI—zero cost, full privacy
- **HuggingFace Spaces Adapter**: Test Gradio-based agents hosted on HuggingFace Spaces
- New environment variables: `EVAL_PROVIDER=huggingface`, `HF_TOKEN`, `EVAL_MODEL`
- Quick start guide for HuggingFace users (`docs/QUICKSTART_HUGGINGFACE.md`)

#### Developer Experience Improvements
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

[Unreleased]: https://github.com/hidai25/eval-view/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/hidai25/eval-view/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/hidai25/eval-view/releases/tag/v0.1.0
