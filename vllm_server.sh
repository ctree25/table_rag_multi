#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
MAX_MODEL_LEN="${VLLM_CONTEXT_LIMIT:-65536}"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}'