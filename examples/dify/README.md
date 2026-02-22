# Dify Example — Testing Dify AI Workflows with EvalView

> Test Dify AI agent workflows with EvalView — verify tool execution, output quality, and detect regressions in visual workflow configurations.

## Setup

### Option A: Dify Cloud (Fastest)

1. Sign up at https://cloud.dify.ai/
2. Create an Agent or Workflow
3. Get your API key from Settings → API Access
4. Use the API endpoint provided

### Option B: Self-Hosted Dify

```bash
# Clone Dify
git clone https://github.com/langgenius/dify.git
cd dify/docker

# Start with Docker Compose
docker compose up -d
```

Access at: `http://localhost:3000`

### 2. Create an Agent in Dify

1. Go to Studio → Create App → Agent
2. Configure your agent with tools
3. Publish the agent
4. Get API endpoint from "Access API"

### 3. Configure Environment

```bash
export DIFY_API_KEY=app-xxxxx
export DIFY_API_URL=https://api.dify.ai/v1  # or your self-hosted URL
```

### 4. Run EvalView Test

```bash
evalview run --pattern examples/dify/test-case.yaml
```

## Links

- **Repo**: https://github.com/langgenius/dify
- **Cloud**: https://cloud.dify.ai/
- **Docs**: https://docs.dify.ai/
- **API Reference**: https://docs.dify.ai/guides/application-publishing/developing-with-apis
