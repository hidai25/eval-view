# CrewAI Example

Test a CrewAI multi-agent content team with EvalView.

## Setup

### 1. Install CrewAI

```bash
pip install crewai crewai-tools
```

### 2. Clone CrewAI Examples

```bash
git clone https://github.com/crewAIInc/crewAI-examples.git
cd crewAI-examples
```

### 3. Run Example Crew

```bash
# Navigate to an example
cd crewAI-examples/stock_analysis
# or
cd crewAI-examples/write_a_book_with_flows

# Run with API server
python main.py --serve
```

Agent will be available at: `http://localhost:8000`

### 4. Run EvalView Test

```bash
# From EvalView root
evalview run --pattern examples/crewai/test-case.yaml
```

## Links

- **Repo**: https://github.com/crewAIInc/crewAI
- **Examples**: https://github.com/crewAIInc/crewAI-examples
- **Docs**: https://docs.crewai.com/
