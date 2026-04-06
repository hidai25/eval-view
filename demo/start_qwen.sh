#!/usr/bin/env bash
# Start llama-server with Qwen3-Coder 30B for OpenCode
# Run this before executing demo/tests/qwen/ tests

MODEL=$(ls "$HOME/models/"*Qwen3-Coder*.gguf 2>/dev/null | head -1)
PORT=8082

if [ -z "$MODEL" ]; then
    echo "Model not found in ~/models/"
    echo "Still downloading? Check: tail -f /tmp/qwen-dl.log"
    exit 1
fi

echo "Starting Qwen3-Coder 30B on port $PORT..."
echo "Model: $MODEL"
echo "Press Ctrl+C to stop"
echo ""

llama-server \
  --model "$MODEL" \
  --port $PORT \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --log-disable
