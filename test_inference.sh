#!/usr/bin/env bash
#
# test_inference.sh — Simulate validator environment locally
#
# Prerequisites:
#   1. Start a LiteLLM proxy:
#      litellm --model huggingface/Qwen/Qwen2.5-72B-Instruct --port 4000
#
#   2. Or use HF router directly (set HF_TOKEN below)
#
# Usage:
#   ./test_inference.sh                  # uses LiteLLM proxy on localhost:4000
#   ./test_inference.sh --hf             # uses HF router directly
#

set -euo pipefail

MODE="${1:-litellm}"

if [ "$MODE" = "--hf" ]; then
    # Direct HF router (no local proxy needed)
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="${ENV_HF_TOKEN:-$(grep -s 'ENV_HF_TOKEN=' .env 2>/dev/null | cut -d= -f2 | tr -d \"\'  || echo '')}"
    if [ -z "$HF_TOKEN" ]; then
        echo "ERROR: Set ENV_HF_TOKEN in .env or environment"
        exit 1
    fi
else
    # LiteLLM proxy (default — mimics validator setup)
    export API_BASE_URL="https://litellm.teachafy.com"
    export MODEL_NAME="gpt-5.4-mini"
    export HF_TOKEN="sk-BkzawtYwKapx1FdZcuz7fg"
fi

# Docker image — pulls from HF registry
export IMAGE_NAME="registry.hf.space/teachafy-cost-aware-finqa:latest"

# Unset any variables that could conflict
unset API_KEY 2>/dev/null || true
unset ENV_HF_TOKEN 2>/dev/null || true

echo "=== Validator Simulation ==="
echo "API_BASE_URL: $API_BASE_URL"
echo "MODEL_NAME:   $MODEL_NAME"
echo "HF_TOKEN:     ${HF_TOKEN:0:8}..."
echo "IMAGE_NAME:   $IMAGE_NAME"
echo "==========================="
echo ""

python3 inference.py
