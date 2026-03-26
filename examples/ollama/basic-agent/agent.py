# examples/ollama/basic-agent/agent.py

import requests
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"


def run_agent(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"error: {e}"


class AgentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        prompt = body.get("query", body.get("input", ""))
        output = run_agent(prompt)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"output": output}).encode())

    def log_message(self, format, *args):
        pass  # silence request logs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "serve":
        print(run_agent(" ".join(sys.argv[1:])))
    else:
        print("Starting agent server on http://localhost:8123")
        HTTPServer(("localhost", 8123), AgentHandler).serve_forever()
