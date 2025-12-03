/**
 * EvalView Middleware for Node.js/Next.js
 * 
 * Drop-in middleware that translates EvalView standard format
 * to your agent's API format and back.
 * 
 * Usage:
 *   import { createEvalViewMiddleware } from '@evalview/node'
 *   
 *   app.post('/api/evalview', createEvalViewMiddleware({
 *     targetEndpoint: '/api/unifiedchat',
 *     parseResponse: (ndjson) => ({ output, steps, cost, latency })
 *   }))
 */

const DEBUG = process.env.EVALVIEW_DEBUG === 'true';

export interface EvalViewRequest {
  query: string;
  context?: {
    route?: string;
    userId?: string;
    conversationId?: string;
    history?: any[];
    [key: string]: any;
  };
  enable_tracing?: boolean;
}

export interface EvalViewStep {
  id: string;
  name: string;
  tool: string;
  parameters: any;
  output?: any;
  success: boolean;
  latency?: number;
  cost?: number;
  tokens?: number;
}

export interface EvalViewResponse {
  session_id: string;
  output: string;
  steps: EvalViewStep[];
  cost: number;
  latency: number;
  tokens?: number;
}

export interface MiddlewareConfig {
  /** Endpoint to forward requests to (e.g., '/api/unifiedchat') */
  targetEndpoint: string;

  /** Optional: Transform EvalView request to your API format */
  transformRequest?: (req: EvalViewRequest) => any;

  /** Optional: Parse your API response to EvalView format */
  parseResponse?: (responseText: string, startTime: number) => EvalViewResponse;

  /** Optional: Base URL for requests (defaults to current host) */
  baseUrl?: string;

  /** Optional: Default user ID for test requests (defaults to 'evalview-test-user') */
  defaultUserId?: string;

  /** Optional: Function to get/create user ID dynamically */
  getUserId?: (req: EvalViewRequest) => string | Promise<string>;
}

/**
 * Default Tapescope request transformer
 */
function defaultTransformRequest(req: EvalViewRequest) {
  const timestamp = Date.now();
  return {
    message: req.query,
    userId: req.context?.userId, // Set by middleware
    conversationId: req.context?.conversationId || `eval-${timestamp}`,
    route: req.context?.route || 'orchestrator',
    history: req.context?.history || [],
  };
}

/**
 * Default Tapescope NDJSON response parser
 * Handles both pure NDJSON and SSE format (with "data: " prefix)
 */
function defaultParseResponse(ndjsonText: string, startTime: number): EvalViewResponse {
  const lines = ndjsonText.trim().split('\n');
  const events = lines
    .filter(line => line.trim())
    .map(line => {
      try {
        // Strip SSE "data: " prefix if present
        const jsonLine = line.startsWith('data: ') ? line.slice(6) : line;
        return JSON.parse(jsonLine);
      } catch (e) {
        return null;
      }
    })
    .filter(Boolean);

  const steps: EvalViewStep[] = [];
  let finalOutput = '';
  let totalCost = 0;
  let totalTokens = 0;
  const sessionId = `session-${startTime}`;

  for (const event of events) {
    const eventType = event.type;

    // Capture tool calls as steps
    if (eventType === 'tool_call') {
      steps.push({
        id: `step-${steps.length}`,
        name: event.tool || 'unknown',
        tool: event.tool || 'unknown',
        parameters: event.params || {},
        output: event.result,
        success: true,
        latency: 0,
        cost: 0,
      });
    }

    // Capture final message
    if (eventType === 'message_complete') {
      finalOutput = event.final?.text || '';
      totalCost = event.data?.metadata?.cost || 0;
      
      // Add synthesis step if present
      if (event.tool) {
        steps.push({
          id: `step-${steps.length}`,
          name: event.tool,
          tool: event.tool,
          parameters: {},
          success: event.ok || false,
          latency: 0,
          cost: totalCost,
        });
      }
    }

    // Accumulate streaming text (fallback if final not set)
    if (eventType === 'text_delta' && !finalOutput) {
      finalOutput += event.delta || '';
    }
  }

  const endTime = Date.now();
  const latency = endTime - startTime;

  return {
    session_id: sessionId,
    output: finalOutput,
    steps,
    cost: totalCost,
    latency,
    tokens: totalTokens,
  };
}

/**
 * Create EvalView middleware handler
 *
 * @param config - Middleware configuration
 * @returns Request handler function
 */
export function createEvalViewMiddleware(config: MiddlewareConfig) {
  const {
    targetEndpoint,
    transformRequest = defaultTransformRequest,
    parseResponse = defaultParseResponse,
    baseUrl,
    defaultUserId = 'evalview-test-user',
    getUserId,
  } = config;

  return async function evalViewHandler(req: any) {
    const startTime = Date.now();

    try {
      // Parse EvalView request (Next.js App Router)
      const evalViewReq: EvalViewRequest = await req.json();

      // Resolve user ID (priority: context.userId > getUserId() > defaultUserId)
      if (!evalViewReq.context) {
        evalViewReq.context = {};
      }
      if (!evalViewReq.context.userId) {
        evalViewReq.context.userId = getUserId
          ? await getUserId(evalViewReq)
          : defaultUserId;
      }

      if (DEBUG) console.log('[EvalView] Received request:', {
        query: evalViewReq.query,
        route: evalViewReq.context?.route,
      });

      // Transform to target API format
      const targetReq = transformRequest(evalViewReq);
      if (DEBUG) console.log('[EvalView] Calling target endpoint:', targetEndpoint);

      // Determine base URL
      const host = req.headers?.get?.('host') || req.headers?.host || 'localhost:3000';
      const requestBaseUrl = baseUrl || `http://${host}`;

      // Call target endpoint
      const response = await fetch(`${requestBaseUrl}${targetEndpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(targetReq),
      });

      if (!response.ok) {
        throw new Error(`Target endpoint failed: ${response.status} ${response.statusText}`);
      }

      // Get response text
      const responseText = await response.text();
      if (DEBUG) console.log('[EvalView] Received response, parsing...');

      // Parse and translate response
      const evalViewRes = parseResponse(responseText, startTime);
      if (DEBUG) console.log('[EvalView] Translated response:', {
        steps: evalViewRes.steps.length,
        outputLength: evalViewRes.output.length,
        cost: evalViewRes.cost,
        latency: evalViewRes.latency,
      });

      // Return JSON response (Next.js App Router format)
      return new Response(JSON.stringify(evalViewRes), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (error: any) {
      console.error('[EvalView] Error:', error);
      return new Response(
        JSON.stringify({
          error: error.message || 'EvalView middleware failed',
          session_id: `error-${startTime}`,
          output: '',
          steps: [],
          cost: 0,
          latency: Date.now() - startTime,
        }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  };
}
