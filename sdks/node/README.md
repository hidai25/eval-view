# @evalview/node

Drop-in Node.js/Next.js middleware for [EvalView](https://github.com/hidai25/eval-view) testing framework.

## Installation

```bash
npm install @evalview/node
```

## Quick Start

### Next.js App Router

```typescript
// app/api/evalview/route.ts
import { createEvalViewMiddleware } from '@evalview/node';

export const POST = createEvalViewMiddleware({
  targetEndpoint: '/api/unifiedchat',  // Your agent's endpoint
});
```

### Express.js

```javascript
const { createEvalViewMiddleware } = require('@evalview/node');

app.post('/api/evalview', createEvalViewMiddleware({
  targetEndpoint: '/api/your-agent',
}));
```

## Configuration

```typescript
createEvalViewMiddleware({
  // Required: Endpoint to forward requests to
  targetEndpoint: '/api/unifiedchat',

  // Optional: Default user ID for test requests (defaults to 'evalview-test-user')
  // Use an existing user ID from your database
  defaultUserId: 'your-dev-user-id',

  // Optional: Dynamic user ID resolution
  // Useful for creating users on-the-fly or looking up existing ones
  getUserId: async (req) => {
    // Example: Look up or create user
    const user = await findOrCreateUser('test@example.com');
    return user.id;
  },

  // Optional: Transform EvalView request to your API format
  transformRequest: (req) => ({
    message: req.query,
    userId: req.context?.userId, // Automatically set by middleware
    // ... your custom mapping
  }),

  // Optional: Parse your API response to EvalView format
  parseResponse: (responseText, startTime) => ({
    session_id: `session-${startTime}`,
    output: '...',
    steps: [...],
    cost: 0.05,
    latency: Date.now() - startTime,
  }),

  // Optional: Base URL for requests
  baseUrl: process.env.API_BASE_URL,
});
```

## Default Behavior

Works out-of-the-box with Tapescope-style APIs that:
- Accept: `{ message, userId, conversationId, route, history }`
- Return: NDJSON stream with `tool_call` and `message_complete` events

## Testing

Point EvalView CLI to your endpoint:

```yaml
# .evalview/config.yaml
adapter: http
endpoint: http://localhost:3000/api/evalview
```

Then run tests:
```bash
evalview run
```

## License

MIT
