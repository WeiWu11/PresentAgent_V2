#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a
source "$ENV_FILE"
set +a

if [ "$MODEL_PATH" = "/your/model/path" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH not configured in .env file"
    exit 1
fi

echo "Starting VLLM servers..."
VLLM_GPU_ID="${VLLM_GPU_ID:-0}"
VLLM_PORT="${VLLM_PORT:-6001}"
echo "Starting a single VLLM server on GPU ${VLLM_GPU_ID} (port ${VLLM_PORT})..."
exec env CUDA_VISIBLE_DEVICES="$VLLM_GPU_ID" \
    vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.65 \
    --max-num-seqs 16 \
    --disable-log-requests
