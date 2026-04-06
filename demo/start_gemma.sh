#!/usr/bin/env bash
# Start llama-server with Gemma 4 26B for OpenCode
# Run this before executing demo/tests/gemma/ tests

MODEL="$HOME/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
PORT=8081

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo "Still downloading? Check: tail -f /tmp/gemma-dl.log"
    exit 1
fi

echo "Starting Gemma 4 26B on port $PORT..."
echo "Context: 32768 tokens (minimum for OpenCode tool definitions)"
echo "Press Ctrl+C to stop"
echo ""

llama-server \
  --model "$MODEL" \
  --port $PORT \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --log-disable
