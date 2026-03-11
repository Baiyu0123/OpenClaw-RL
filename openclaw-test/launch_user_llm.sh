#!/usr/bin/env bash
#
# Launch an SGLang API server for Qwen3-4B.
#
# Required environment variables:
#   MODEL_PATH      - Absolute path to the Qwen3-4B model weights directory
#
# Optional environment variables:
#   HOST            - Bind address       (default: 0.0.0.0)
#   PORT            - Listen port        (default: 30001)
#   TP_SIZE         - Tensor parallel    (default: 8)
#   MAX_TOKENS      - Max total tokens   (default: 32768)
#   MODEL_NAME      - served-model-name  (default: qwen3-4b-user-llm)
#   SGLANG_API_KEY  - API key for auth   (default: none, no auth)
#
# Usage:
#   export MODEL_PATH="/data/models/Qwen/Qwen3-4B"
#   bash launch_user_llm.sh

set -euo pipefail

if [ -z "${MODEL_PATH:-}" ]; then
    echo "Error: MODEL_PATH is not set." >&2
    echo "Usage: MODEL_PATH=/path/to/Qwen3-4B bash $0" >&2
    exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30001}"
TP_SIZE="${TP_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b-user-llm}"
API_KEY="${SGLANG_API_KEY:-}"

API_KEY_ARGS=()
if [ -n "${API_KEY}" ]; then
    API_KEY_ARGS=(--api-key "${API_KEY}")
fi

echo "============================================"
echo "  SGLang API Server"
echo "  Model:  ${MODEL_PATH}"
echo "  Host:   ${HOST}:${PORT}"
echo "  TP:     ${TP_SIZE}"
echo "============================================"

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --context-length "${MAX_TOKENS}" \
    --served-model-name "${MODEL_NAME}" \
    "${API_KEY_ARGS[@]}"
