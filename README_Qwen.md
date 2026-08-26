export VLLM_CONTEXT_LIMIT=16384

CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len "$VLLM_CONTEXT_LIMIT" \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}'