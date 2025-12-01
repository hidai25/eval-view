"""
HTTP server wrapper for Claude Agent SDK agent.

This allows EvalView to test the agent via the HTTP adapter.

Usage:
    python server.py

Then run:
    evalview run examples/claude-agent-sdk
"""

import anyio
from flask import Flask, request, jsonify
from agent import run_agent

app = Flask(__name__)


@app.route("/agent", methods=["POST"])
def agent_endpoint():
    """Handle agent requests from EvalView."""
    data = request.json
    query = data.get("query", data.get("input", data.get("message", "")))

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Run the agent
    try:
        response = anyio.run(run_agent, query)

        # Return in EvalView-compatible format
        return jsonify({
            "output": response,
            "steps": [],  # Tool calls are handled internally by SDK
            "metrics": {
                "latency": 0,
                "cost": 0,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting Claude Agent SDK server on http://localhost:5001")
    print("Test with: curl -X POST http://localhost:5001/agent -H 'Content-Type: application/json' -d '{\"query\": \"Hello\"}'")
    app.run(host="0.0.0.0", port=5001, debug=True)
