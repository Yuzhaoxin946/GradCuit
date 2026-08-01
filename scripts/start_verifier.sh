set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"

MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
SERVED_MODEL_NAME="Llama-3.2-3B-Instruct"
PORT="8000"

vllm serve "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "127.0.0.1" \
    --port "$PORT" \
    --dtype "bfloat16" \
    --max-model-len "8192" \
    --gpu-memory-utilization "0.85" \
    --generation-config "vllm"
