# Regression Testing CrewAI Agents with EvalView

**The problem:** You changed a prompt, swapped a model, or updated a tool — and `crewai test` says your scores look fine. But the researcher agent stopped calling the search tool, the analyst skipped the calculator, and the final report is subtly worse. Scores don't catch this. **Tool-level diffs do.**

EvalView snapshots your crew's full execution trace — which agent called which tool, in what order, with what parameters — and diffs it against a baseline on every change. When an agent silently changes behavior, you see exactly what shifted.

## Quick Start (5 minutes)

### 1. Install

```bash
pip install evalview crewai crewai-tools
```

### 2. Start your crew's API server

```bash
cd your-crew-project
crewai run --serve
# Server at http://localhost:8000
```

### 3. Point EvalView at it

```bash
evalview init
# Select "crewai" adapter, enter http://localhost:8000/crew
```

### 4. Capture a baseline

```bash
evalview snapshot
```

EvalView runs your test queries, records every agent step (tool calls, parameters, outputs, cost), and saves it as a golden baseline.

### 5. Make a change, catch the diff

```bash
# Edit a prompt, swap a model, update a tool...
evalview check
```

```
  ✓ stock-analysis           PASSED
  ⚠ content-team             TOOLS_CHANGED
      Step 2: analyst_agent
      - calculator_tool(ticker="AAPL", metric="pe_ratio")
      + calculator_tool(ticker="AAPL", metric="market_cap")
  ✗ trip-planner             REGRESSION  -25 pts
      researcher_agent skipped search_tool entirely
      Score: 85 → 60  Output similarity: 42%
```

That's what `crewai test` doesn't show you.

## Test Case Examples

### Single crew test

```yaml
# tests/stock-analysis.yaml
name: stock-analysis
adapter: crewai
endpoint: http://localhost:8000/crew

input:
  query: "Analyze Apple (AAPL) stock for investment potential"
  context:
    ticker: "AAPL"
    analysis_type: "comprehensive"

expected:
  tools:
    - ScrapeWebsiteTool
    - WebsiteSearchTool
    - CalculatorTool
  output:
    contains:
      - "Apple"
      - "AAPL"
    not_contains:
      - "error"

thresholds:
  min_score: 70
  max_cost: 1.00
  max_latency: 120000
```

### Multi-agent delegation test

```yaml
# tests/content-team.yaml
name: content-team
adapter: crewai
endpoint: http://localhost:8000/crew

input:
  query: "Write a blog post about AI agents in healthcare"
  context:
    topic: "AI agents"
    industry: "healthcare"
    length: "1000 words"

expected:
  tools:
    - WebsiteSearchTool
    - ScrapeWebsiteTool
  output:
    contains:
      - "healthcare"
      - "AI"
    not_contains:
      - "error"
      - "I cannot"

thresholds:
  min_score: 70
  max_cost: 0.50
  max_latency: 120000
```

### Safety-critical crew (forbidden tools)

```yaml
# tests/customer-support-crew.yaml
name: customer-support-crew
adapter: crewai
endpoint: http://localhost:8000/crew

input:
  query: "Customer wants to cancel their account"

expected:
  tools:
    - lookup_customer
    - check_retention_offers
  forbidden_tools:
    - delete_account
    - process_refund
  output:
    contains:
      - "retention"
      - "offer"

thresholds:
  min_score: 75
```

## CI Integration

Add to your CrewAI project's GitHub Actions:

```yaml
# .github/workflows/evalview.yml
name: EvalView Agent Check
on: [pull_request]

jobs:
  agent-check:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4

      - name: Start crew server
        run: |
          pip install crewai crewai-tools
          crewai run --serve &
          sleep 10  # Wait for server startup

      - name: Check for regressions
        uses: hidai25/eval-view@v0.8.0
        with:
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
```

Every PR gets a comment showing what changed in your crew's behavior.

## Watch Mode

Leave it running while you iterate on prompts:

```bash
evalview watch --quick    # Re-checks on every file save, $0, sub-second
```

## Configuration

```yaml
# .evalview/config.yaml
adapter: crewai
endpoint: http://localhost:8000/crew
timeout: 120  # CrewAI crews can take 1-2 minutes
```

## What EvalView Catches That `crewai test` Doesn't

| | `crewai test` | EvalView |
|---|---|---|
| Score per task | Yes (1-10) | Yes (0-100) |
| Which tools each agent called | No | Yes |
| Tool parameter changes | No | Yes |
| Agent delegation order changes | No | Yes |
| Output similarity vs baseline | No | Yes |
| Cost/latency tracking | No | Yes |
| CI/PR comments | No | Yes |
| Forbidden tool violations | No | Yes |

## Troubleshooting

**"Connection refused on port 8000"**
- Make sure the crew is running with `--serve` flag
- Some examples need a custom FastAPI wrapper — see CrewAI docs

**"Crew takes too long"**
- Increase timeout: `timeout: 180` in config
- Multi-agent crews with search tools can take 2+ minutes

**"Missing API keys"**
- Check the crew's README for required keys
- Common: `OPENAI_API_KEY`, `SERPER_API_KEY`

## Links

- [CrewAI Docs](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [CrewAI Examples](https://github.com/crewAIInc/crewAI-examples)
- [EvalView Framework Support](../../docs/FRAMEWORK_SUPPORT.md)
