"""MCP (Model Context Protocol) adapter for testing MCP servers.

MCP is Anthropic's protocol for connecting AI agents to external tools.
This adapter allows testing MCP servers directly, validating that tools
work correctly before deploying them with Claude Desktop, Claude Code, etc.

Requirements:
    - MCP server running (stdio or HTTP transport)
    - pip install mcp (optional, for type hints)

Usage in test case:
    adapter: mcp
    endpoint: "stdio:python my_server.py"  # or "http://localhost:8080"

    input:
      query: "read_file"  # Tool name to test
      context:
        arguments:
          path: "/tmp/test.txt"

Or test a sequence of tool calls:
    input:
      query: "multi"
      context:
        tool_calls:
          - tool: read_file
            arguments: { path: "/tmp/test.txt" }
          - tool: write_file
            arguments: { path: "/tmp/out.txt", content: "hello" }
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from evalview.adapters.base import AgentAdapter
from evalview.core.types import (
    ExecutionMetrics,
    ExecutionTrace,
    StepMetrics,
    StepTrace,
    SpanKind,
)
from evalview.core.tracing import Tracer

logger = logging.getLogger(__name__)


class MCPAdapter(AgentAdapter):
    """Adapter for testing MCP (Model Context Protocol) servers.

    Supports two transport modes:
    - stdio: Spawn server as subprocess, communicate via stdin/stdout
    - http: Connect to HTTP server endpoint

    This adapter lets you test MCP tools directly without needing
    a full AI agent, making it easy to validate tool implementations.
    """

    def __init__(
        self,
        endpoint: str = "",
        timeout: float = 30.0,
        **kwargs: Any,
    ):
        """Initialize MCP adapter.

        Args:
            endpoint: Server endpoint in format:
                - "stdio:command args" - spawn subprocess
                - "http://host:port" - connect to HTTP server
                - "npx:@modelcontextprotocol/server-filesystem" - npm package
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self._request_id = 0

    @property
    def name(self) -> str:
        return "mcp"

    def _parse_endpoint(self) -> Tuple[str, str]:
        """Parse endpoint into (transport, target)."""
        if self.endpoint.startswith("stdio:"):
            return ("stdio", self.endpoint[6:])
        elif self.endpoint.startswith("npx:"):
            return ("stdio", f"npx -y {self.endpoint[4:]}")
        elif self.endpoint.startswith("http://") or self.endpoint.startswith("https://"):
            return ("http", self.endpoint)
        else:
            # Default to stdio with command
            return ("stdio", self.endpoint)

    async def execute(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """Execute MCP tool(s) and capture trace.

        Args:
            query: Tool name to execute, or "multi" for multiple tools
            context: Tool arguments and options:
                - arguments: Dict of tool arguments (for single tool)
                - tool_calls: List of {tool, arguments} for multi-tool

        Returns:
            ExecutionTrace with tool results
        """
        context = context or {}
        session_id = f"mcp-{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        steps: List[StepTrace] = []

        # Initialize tracer
        tracer = Tracer()

        transport, target = self._parse_endpoint()

        try:
            async with tracer.start_span_async("MCP Server", SpanKind.AGENT):
                if transport == "stdio":
                    steps = await self._execute_stdio(query, context, target, tracer)
                else:
                    steps = await self._execute_http(query, context, target, tracer)

            end_time = datetime.now()

            # Build final output from step results
            final_output = self._build_output(steps)

            return ExecutionTrace(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                steps=steps,
                final_output=final_output,
                metrics=ExecutionMetrics(
                    total_cost=0.0,  # MCP tools don't have LLM cost
                    total_latency=(end_time - start_time).total_seconds() * 1000,
                    total_tokens=None,
                ),
                trace_context=tracer.build_trace_context(),
            )

        except Exception as e:
            end_time = datetime.now()
            logger.error(f"MCP execution failed: {e}")
            return self._create_error_trace(str(e), start_time, end_time, tracer)

    async def _execute_stdio(
        self, query: str, context: Dict[str, Any], command: str, tracer: Tracer
    ) -> List[StepTrace]:
        """Execute via stdio transport (subprocess)."""
        steps = []

        # Parse command
        import shlex
        cmd_parts = shlex.split(command)

        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Initialize MCP session
            init_result = await self._send_request(
                process,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "evalview",
                        "version": "0.1.7"
                    }
                }
            )

            if not init_result.get("success", True):
                raise Exception(f"MCP init failed: {init_result}")

            # Send initialized notification
            await self._send_notification(process, "notifications/initialized", {})

            # Get tool calls to execute
            if query == "multi":
                tool_calls = context.get("tool_calls", [])
            else:
                tool_calls = [{
                    "tool": query,
                    "arguments": context.get("arguments", {})
                }]

            # Execute each tool
            for i, call in enumerate(tool_calls):
                tool_name = call.get("tool", call.get("name", "unknown"))
                arguments = call.get("arguments", call.get("params", {}))

                step_start = datetime.now()

                result = await self._send_request(
                    process,
                    "tools/call",
                    {
                        "name": tool_name,
                        "arguments": arguments
                    }
                )

                step_end = datetime.now()
                step_latency = (step_end - step_start).total_seconds() * 1000

                # Parse result
                success = not result.get("isError", False)
                content = result.get("content", [])
                output = self._format_content(content)
                error = result.get("error") if not success else None

                # Record tool span
                tracer.record_tool_call(
                    tool_name=tool_name,
                    parameters=arguments,
                    result=output,
                    error=error,
                    duration_ms=step_latency,
                )

                steps.append(StepTrace(
                    step_id=f"step-{i+1}",
                    step_name=f"Call {tool_name}",
                    tool_name=tool_name,
                    parameters=arguments,
                    output=output,
                    success=success,
                    error=error,
                    metrics=StepMetrics(latency=step_latency, cost=0.0),
                ))

        finally:
            # Clean shutdown
            process.terminate()
            await process.wait()

        return steps

    async def _execute_http(
        self, query: str, context: Dict[str, Any], url: str, tracer: Tracer
    ) -> List[StepTrace]:
        """Execute via HTTP transport."""
        import httpx

        steps = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Get tool calls
            if query == "multi":
                tool_calls = context.get("tool_calls", [])
            else:
                tool_calls = [{
                    "tool": query,
                    "arguments": context.get("arguments", {})
                }]

            for i, call in enumerate(tool_calls):
                tool_name = call.get("tool", call.get("name", "unknown"))
                arguments = call.get("arguments", call.get("params", {}))

                step_start = datetime.now()

                # Send JSON-RPC request
                self._request_id += 1
                response = await client.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "id": self._request_id,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    }
                )

                step_end = datetime.now()
                step_latency = (step_end - step_start).total_seconds() * 1000

                result = response.json()

                if "error" in result:
                    error_msg = str(result["error"])
                    # Record tool span for error
                    tracer.record_tool_call(
                        tool_name=tool_name,
                        parameters=arguments,
                        result="",
                        error=error_msg,
                        duration_ms=step_latency,
                    )
                    steps.append(StepTrace(
                        step_id=f"step-{i+1}",
                        step_name=f"Call {tool_name}",
                        tool_name=tool_name,
                        parameters=arguments,
                        output="",
                        success=False,
                        error=error_msg,
                        metrics=StepMetrics(latency=step_latency, cost=0.0),
                    ))
                else:
                    content = result.get("result", {}).get("content", [])
                    output = self._format_content(content)

                    # Record tool span for success
                    tracer.record_tool_call(
                        tool_name=tool_name,
                        parameters=arguments,
                        result=output,
                        error=None,
                        duration_ms=step_latency,
                    )

                    steps.append(StepTrace(
                        step_id=f"step-{i+1}",
                        step_name=f"Call {tool_name}",
                        tool_name=tool_name,
                        parameters=arguments,
                        output=output,
                        success=True,
                        error=None,
                        metrics=StepMetrics(latency=step_latency, cost=0.0),
                    ))

        return steps

    async def _send_request(
        self, process: asyncio.subprocess.Process, method: str, params: Dict
    ) -> Dict:
        """Send JSON-RPC request via stdio."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params
        }

        # Write request
        request_line = json.dumps(request) + "\n"
        process.stdin.write(request_line.encode())
        await process.stdin.drain()

        # Read response
        response_line = await asyncio.wait_for(
            process.stdout.readline(),
            timeout=self.timeout
        )

        if not response_line:
            raise Exception("No response from MCP server")

        response = json.loads(response_line.decode())

        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")

        return response.get("result", {})

    async def _send_notification(
        self, process: asyncio.subprocess.Process, method: str, params: Dict
    ) -> None:
        """Send JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        notification_line = json.dumps(notification) + "\n"
        process.stdin.write(notification_line.encode())
        await process.stdin.drain()

    def _format_content(self, content: List[Dict]) -> str:
        """Format MCP content array to string."""
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
            elif item.get("type") == "resource":
                parts.append(f"[Resource: {item.get('uri', 'unknown')}]")
            else:
                parts.append(str(item))
        return "\n".join(parts)

    def _build_output(self, steps: List[StepTrace]) -> str:
        """Build final output from step results."""
        parts = []
        for step in steps:
            if step.success:
                parts.append(f"[{step.tool_name}] {step.output}")
            else:
                parts.append(f"[{step.tool_name}] ERROR: {step.error}")
        return "\n\n".join(parts)

    def _create_error_trace(
        self, error_msg: str, start_time: datetime, end_time: datetime, tracer: Tracer
    ) -> ExecutionTrace:
        """Create error trace."""
        # Record the error as a tool call
        tracer.record_tool_call(
            tool_name="error",
            parameters={},
            result=error_msg,
            error=error_msg,
            duration_ms=(end_time - start_time).total_seconds() * 1000,
        )

        return ExecutionTrace(
            session_id=f"mcp-error-{uuid.uuid4().hex[:8]}",
            start_time=start_time,
            end_time=end_time,
            steps=[
                StepTrace(
                    step_id="error",
                    step_name="Error",
                    tool_name="error",
                    parameters={},
                    output=error_msg,
                    success=False,
                    error=error_msg,
                    metrics=StepMetrics(),
                )
            ],
            final_output=error_msg,
            metrics=ExecutionMetrics(
                total_cost=0.0,
                total_latency=(end_time - start_time).total_seconds() * 1000,
                total_tokens=None,
            ),
            trace_context=tracer.build_trace_context(),
        )

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from an MCP server.

        Connects to the server, initializes the session, and calls tools/list
        to retrieve all available tool definitions (name, description, inputSchema).

        Returns:
            List of tool definition dicts from the MCP server.

        Raises:
            Exception: If connection or discovery fails.
        """
        transport, target = self._parse_endpoint()

        if transport == "stdio":
            return await self._discover_tools_stdio(target)
        else:
            return await self._discover_tools_http(target)

    async def _discover_tools_stdio(self, command: str) -> List[Dict[str, Any]]:
        """Discover tools via stdio transport."""
        import shlex

        cmd_parts = shlex.split(command)
        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await self._send_request(
                process,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "evalview", "version": "0.1.7"},
                },
            )
            await self._send_notification(process, "notifications/initialized", {})

            result = await self._send_request(process, "tools/list", {})
            return result.get("tools", [])
        finally:
            process.terminate()
            await process.wait()

    async def _discover_tools_http(self, url: str) -> List[Dict[str, Any]]:
        """Discover tools via HTTP transport."""
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Initialize session
            self._request_id += 1
            await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "evalview", "version": "0.1.7"},
                    },
                },
            )

            # Request tool list
            self._request_id += 1
            response = await client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "tools/list",
                    "params": {},
                },
            )
            result = response.json()

            if "error" in result:
                raise Exception(f"MCP tools/list error: {result['error']}")

            return result.get("result", {}).get("tools", [])

    async def health_check(self) -> bool:
        """Check if MCP server is reachable."""
        transport, target = self._parse_endpoint()

        if transport == "http":
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        target,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "evalview", "version": "0.1.7"}
                            }
                        }
                    )
                    return response.status_code == 200
            except Exception:
                return False
        else:
            # For stdio, just check command exists
            import shlex
            cmd = shlex.split(target)[0]
            import shutil
            return shutil.which(cmd) is not None
