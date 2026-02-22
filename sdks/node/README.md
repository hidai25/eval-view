# @evalview/node — Proof that your agent still works.

> You changed a prompt. Swapped a model. Updated a tool.
> Did anything break? **Run EvalView. Know for sure.**

Drop-in Node.js/Next.js middleware for [EvalView](https://github.com/hidai25/eval-view) — the regression testing framework for AI agents.

---

## What EvalView Catches

| Status | What it means | What you do |
|--------|--------------|-------------|
| ✅ **PASSED** | Agent behavior matches baseline | Ship with confidence |
| ⚠️ **TOOLS_CHANGED** | Agent is calling different tools | Review the diff |
| ⚠️ **OUTPUT_CHANGED** | Same tools, output quality shifted | Review the diff |
| ❌ **REGRESSION** | Score dropped significantly | Fix before shipping |

---

## Quick Start

```bash
npm install @evalview/node
pip install evalview
```

### Next.js App Router

```typescript
// app/api/evalview/route.ts
import { createEvalViewMiddleware } from '@evalview/node';

export const POST = createEvalViewMiddleware({
  targetEndpoint: '/api/your-agent',
});
```

### Express.js

```javascript
const { createEvalViewMiddleware } = require('@evalview/node');

app.post('/api/evalview', createEvalViewMiddleware({
  targetEndpoint: '/api/your-agent',
}));
```

Then point EvalView at your endpoint:

```yaml
# .evalview/config.yaml
adapter: http
endpoint: http://localhost:3000/api/evalview
```

Capture baseline and check for regressions:

```bash
evalview snapshot   # save current behavior as baseline
evalview check      # detect regressions on every change
```

---

## Claude Code Integration (MCP)

Test your agent without leaving the conversation:

```bash
claude mcp add --transport stdio evalview -- evalview mcp serve
cp CLAUDE.md.example CLAUDE.md
```

Ask Claude Code naturally:

```
You: Did my refactor break anything?
Claude: [run_check] ✨ All clean! No regressions detected.

You: Add a test for my weather agent
Claude: [create_test] ✅ Created tests/weather-lookup.yaml
        [run_snapshot] 📸 Baseline captured.
```

No YAML. No terminal switching. No context loss.

[Full MCP docs →](https://github.com/hidai25/eval-view#-claude-code-integration-mcp)

---

## Configuration

```typescript
createEvalViewMiddleware({
  // Required: your agent's endpoint
  targetEndpoint: '/api/your-agent',

  // Optional: default user ID for test requests
  defaultUserId: 'your-dev-user-id',

  // Optional: dynamic user ID resolution
  getUserId: async (req) => {
    const user = await findOrCreateUser('test@example.com');
    return user.id;
  },

  // Optional: transform EvalView request to your API format
  transformRequest: (req) => ({
    message: req.query,
    userId: req.context?.userId,
  }),

  // Optional: parse your API response to EvalView format
  parseResponse: (responseText, startTime) => ({
    session_id: `session-${startTime}`,
    output: '...',
    steps: [...],
    cost: 0.05,
    latency: Date.now() - startTime,
  }),
});
```

---

## Automate It

```yaml
# .github/workflows/evalview.yml
- run: evalview check --fail-on REGRESSION --json
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Documentation

[Full docs →](https://github.com/hidai25/eval-view) • [Examples →](https://github.com/hidai25/eval-view/tree/main/examples) • [Issues →](https://github.com/hidai25/eval-view/issues)

---

## License

Apache-2.0
