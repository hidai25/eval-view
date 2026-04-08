#!/usr/bin/env bash
# Start Gemma 4 26B via Ollama for OpenCode
# Install Ollama: https://ollama.com/download
# Run this before executing demo/tests/gemma/ tests

echo "Pulling Gemma 4 26B via Ollama (skipped if already downloaded)..."
ollama pull gemma4:26b

echo ""
echo "Gemma 4 26B is ready. Ollama serves it automatically on http://localhost:11434"
echo "Run your tests with: evalview run demo/tests/gemma/"
