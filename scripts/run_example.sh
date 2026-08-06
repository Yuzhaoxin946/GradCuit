set -euo pipefail

export CUDA_VISIBLE_DEVICES="3"

python src/main.py \
    --dataset "openai/gsm8k" \
    --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
    --output_dir "./output" \
    --insert_prefix_text "Let's think about this problem and solve it step by step." \
    --solver_prompt_idx "0" \
    --seed "42" \
    --lr "0.001" \
    --optimizer "adam" \
    --max_num_steps "10" \
    --max_new_tokens "2048" \
    --optimize_layer_idx "14" \
    --grad_clip "0" \
    --reward_threshold "-0.2" \
    --start_data_idx "0" \
    --num_data "10" \
    --device "cuda" \
    --vllm_base_url "http://127.0.0.1:8000/v1" \
    --vllm_verifier_model_name "Llama-3.2-3B-Instruct" \
    --vllm_timeout "300"
